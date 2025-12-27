import json
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import glob
from scipy.ndimage import zoom, gaussian_filter


class TrainingAnalyzer:
    def __init__(self, log_pattern='training_metrics_worker_*.jsonl'):
        self.log_pattern = log_pattern
        self.episodes = []
        self._load_all_logs()
    
    def _load_all_logs(self):
        files = sorted(glob.glob(self.log_pattern))
        if not files:
            print(f"No log files found for pattern: {self.log_pattern}")
            return
        total_loaded = 0
        for fpath in files:
            try:
                with open(fpath, 'r') as f:
                    for line in f:
                        self.episodes.append(json.loads(line))
                        total_loaded += 1
            except FileNotFoundError:
                print(f"Could not open {fpath}")
        print(f"Loaded {total_loaded} total episodes from {len(files)} workers")
    
    def plot_winrate_over_time(self, ax, window=50):
        if not self.episodes:
            return
        winrates = []
        episode_nums = []
        for i in range(window, len(self.episodes) + 1):
            chunk = self.episodes[i-window:i]
            wins = sum(1 for e in chunk if e['outcome'] == 'win')
            winrate = (wins / len(chunk)) * 100
            winrates.append(winrate)
            episode_nums.append(i)
        ax.plot(episode_nums, winrates, linewidth=2.5, color='#2ecc71')
        ax.fill_between(episode_nums, winrates, alpha=0.3, color='#2ecc71')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        if len(episode_nums) > 10:
            z = np.polyfit(episode_nums, winrates, 2)
            p = np.poly1d(z)
            ax.plot(episode_nums, p(episode_nums), "--", alpha=0.5, color='darkgreen', linewidth=2, label='Trend')
        ax.set_xlabel('Episode', fontsize=10, fontweight='bold')
        ax.set_ylabel('Winrate (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'Rolling Winrate (window={window})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
    
    def plot_game_length_comparison(self, ax):
        wins = [e for e in self.episodes if e['outcome'] == 'win']
        losses = [e for e in self.episodes if e['outcome'] == 'loss']
        draws = [e for e in self.episodes if e['outcome'] == 'draw']
        data, labels, colors = [], [], []
        if wins:
            data.append([e['game_time_seconds'] for e in wins])
            labels.append(f'Wins\n(n={len(wins)})')
            colors.append('#2ecc71')
        if losses:
            data.append([e['game_time_seconds'] for e in losses])
            labels.append(f'Losses\n(n={len(losses)})')
            colors.append('#e74c3c')
        if draws:
            data.append([e['game_time_seconds'] for e in draws])
            labels.append(f'Draws\n(n={len(draws)})')
            colors.append('#95a5a6')
        if not data:
            return
        bp = ax.boxplot(data, positions=range(1, len(data)+1), widths=0.6, patch_artist=True,
                        showmeans=True, meanline=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
        ax.set_ylabel('Game Time (sec)', fontsize=10, fontweight='bold')
        ax.set_title('Game Duration', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_unit_comparison(self, ax):
        wins = [e for e in self.episodes if e['outcome'] == 'win']
        if not wins:
            ax.text(0.5, 0.5, 'No wins yet', ha='center', va='center')
            return
        unit_order = [
            'scvs', 'mules', 'marines', 'marauders', 'reapers', 'ghosts',
            'hellions', 'hellbats', 'siege_tanks', 'thors', 'cyclones', 'widow_mines',
            'vikings', 'medivacs', 'liberators', 'ravens', 'banshees', 'battlecruisers'
        ]
        averages = []
        labels = []
        for unit in unit_order:
            values = []
            for e in wins:
                if 'peak_units' in e:
                    val = e['peak_units'].get(unit, 0)
                elif 'final_units' in e:
                    val = e['final_units'].get(unit, 0)
                else:
                    val = e.get(unit, 0)
                values.append(val)
            avg = statistics.mean(values) if values else 0
            if avg > 0.5:
                averages.append(avg)
                label = unit.replace('_', ' ').title()
                labels.append(label)
        if not averages:
            ax.text(0.5, 0.5, 'No unit data', ha='center', va='center')
            return
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(averages)))
        bars = ax.bar(labels, averages, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_ylabel('Average Peak Count', fontsize=10, fontweight='bold')
        ax.set_title(f'Peak Units (n={len(wins)} wins)', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_building_heatmap_by_tier(self, ax, heatmap_size=64):
        """
        Plot building placements with RED, BLUE, GREEN colors.
        Uses Simple64 map as background.
        """
        from scipy.ndimage import gaussian_filter
        
        wins = [e for e in self.episodes if e.get('outcome') == 'win' and 'building_placements' in e]
        if not wins:
            ax.text(0.5, 0.5, 'No placement data', ha='center', va='center')
            return
        
        # Define building tier classification
        anchor_buildings = {'command_center', 'orbital_command', 'planetary_fortress'}
        production_buildings = {'barracks', 'factory', 'starport'}
        accessory_buildings = {
            'supply_depot', 'refinery', 'engineering_bay', 'armory', 
            'missile_turret', 'sensor_tower', 'bunker', 'fusion_core',
            'ghost_academy', 'barracks_techlab', 'barracks_reactor',
            'factory_techlab', 'factory_reactor', 'starport_techlab', 'starport_reactor'
        }
        
        background_size = 68
        
        # Load and resize background
        try:
            from PIL import Image
            map_img = Image.open('simple64.png')
            map_img = map_img.resize((background_size, background_size), Image.LANCZOS)
            map_background = np.array(map_img).astype(float) / 255.0
            if map_background.shape[2] == 4:
                map_background = map_background[:, :, :3]
            
            # Brighten background by 1.5x
            map_background = np.clip(map_background * 1.5, 0, 1)
            
            # Desaturate background (reduce color intensity to 70%)
            gray = np.mean(map_background, axis=2, keepdims=True)
            map_background = gray * 0.3 + map_background * 0.7
            
            map_background = np.flipud(map_background)
        except Exception as e:
            print(f"Could not load simple64.png: {e}, using gray background")
            map_background = np.ones((background_size, background_size, 3)) * 0.35
        
        # Separate coordinates by tier
        anchor_coords = []
        production_coords = []
        accessory_coords = []
        
        for ep in wins:
            for placement in ep['building_placements']:
                # Skip starting command center
                if placement.get('type') == 'command_center' and placement.get('step', 0) == 0:
                    continue
                if not placement.get('success', True):
                    continue
                
                x, y = placement['x'], placement['y']
                if x < 0 or x > 1.0 or y < 0 or y > 1.0:
                    continue
                
                # Classify by building type if tier not specified
                building_type = placement.get('type', '').lower()
                tier = placement.get('tier', '').lower()
                
                # Use tier if available, otherwise classify by building type
                if tier == 'anchor' or building_type in anchor_buildings:
                    anchor_coords.append((x, y))
                elif tier == 'production' or building_type in production_buildings:
                    production_coords.append((x, y))
                elif tier == 'accessory' or building_type in accessory_buildings:
                    accessory_coords.append((x, y))
                else:
                    # Default to accessory for unknown buildings
                    accessory_coords.append((x, y))
        
        print(f"Building counts - Anchor: {len(anchor_coords)}, Production: {len(production_coords)}, Accessory: {len(accessory_coords)}")
        
        # Create heatmaps
        def create_heatmap(coords):
            if not coords:
                return np.zeros((heatmap_size, heatmap_size))
            
            coords_array = np.array(coords)
            x_bins = (coords_array[:, 0] * heatmap_size).astype(int)
            y_bins = (coords_array[:, 1] * heatmap_size).astype(int)
            x_bins = np.clip(x_bins, 0, heatmap_size-1)
            y_bins = np.clip(y_bins, 0, heatmap_size-1)
            
            heatmap = np.zeros((heatmap_size, heatmap_size))
            for x, y in zip(x_bins, y_bins):
                heatmap[y, x] += 1
            
            # Apply square root scaling to compress high values
            heatmap = np.sqrt(heatmap)
            
            # Minimal bloom
            heatmap = gaussian_filter(heatmap, sigma=0.8)
            
            # Normalize
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            return heatmap
        
        anchor_heat = create_heatmap(anchor_coords)
        production_heat = create_heatmap(production_coords)
        accessory_heat = create_heatmap(accessory_coords)
        
        # Create RGB heatmap - Red, Blue, Yellow primaries
        rgb_heatmap = np.zeros((heatmap_size, heatmap_size, 3))
        # Red for Anchor
        rgb_heatmap[:, :, 0] += anchor_heat * 2.0
        # Blue for Production
        rgb_heatmap[:, :, 2] += production_heat * 5.0
        # Yellow for Accessory (R + G)
        rgb_heatmap[:, :, 0] += accessory_heat * 5.0
        rgb_heatmap[:, :, 1] += accessory_heat * 5.0
        rgb_heatmap = np.clip(rgb_heatmap, 0, 1)
        
        # Display background
        ax.imshow(map_background, origin='lower', interpolation='bilinear', 
                extent=[0, background_size, 0, background_size])
        
        # Calculate centering offset for heatmap
        offset = (background_size - heatmap_size) / 2
        
        # Overlay heatmap with transparency
        alpha_channel = np.maximum(np.maximum(rgb_heatmap[:,:,0], rgb_heatmap[:,:,1]), rgb_heatmap[:,:,2])
        # Boost alpha significantly for visibility
        alpha_channel = np.clip(alpha_channel * 5.0, 0, 1)
        rgb_heatmap_with_alpha = np.dstack([rgb_heatmap, alpha_channel])
        
        ax.imshow(rgb_heatmap_with_alpha, origin='lower', interpolation='bilinear',
                extent=[offset, offset + heatmap_size, offset, offset + heatmap_size])
        
        ax.set_title('Building Placement Heatmap\n(Red=Anchor, Blue=Production, Yellow=Accessory)', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_xticks([0, background_size/2, background_size-1])
        ax.set_yticks([0, background_size/2, background_size-1])
        ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=7)
        ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=7)
        ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
            
    def plot_economy_progression(self, ax):
        episodes = [e for e in self.episodes if 'building_placements' in e]
        if not episodes:
            ax.text(0.5, 0.5, 'No economy data', ha='center', va='center')
            return
        step_data = defaultdict(lambda: {'minerals': [], 'vespene': [], 'scvs': [], 'army': []})
        for e in episodes:
            for snap in e['economy_snapshots']:
                step = snap['step']
                step_data[step]['minerals'].append(snap['minerals'])
                step_data[step]['vespene'].append(snap['vespene'])
                step_data[step]['scvs'].append(snap['scvs'])
                step_data[step]['army'].append(snap['total_army'])
        steps = sorted(step_data.keys())
        minerals = [statistics.mean(step_data[s]['minerals']) for s in steps]
        vespene = [statistics.mean(step_data[s]['vespene']) for s in steps]
        scvs = [statistics.mean(step_data[s]['scvs']) for s in steps]
        army = [statistics.mean(step_data[s]['army']) for s in steps]
        ax2 = ax.twinx()
        line1 = ax.plot(steps, minerals, label='Minerals', color='#f39c12', linewidth=2)
        line2 = ax.plot(steps, vespene, label='Vespene', color='#16a085', linewidth=2)
        line3 = ax2.plot(steps, scvs, label='SCVs', color='#3498db', linewidth=2)
        line4 = ax2.plot(steps, army, label='Army', color='#e74c3c', linewidth=2)
        ax.set_xlabel('Game Step', fontsize=9, fontweight='bold')
        ax.set_ylabel('Resources', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Units', fontsize=9, fontweight='bold')
        ax.set_title('Economy & Army Over Time (Wins)', fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_action_distribution(self, ax):
        wins = [e for e in self.episodes if e['outcome'] == 'win']
        if not wins:
            return
        action_counts = defaultdict(int)
        for w in wins:
            for action, count in w.get('meaningful_actions', {}).items():
                action_counts[action] += count
        top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:12]
        actions, counts = zip(*top_actions) if top_actions else ([], [])
        if not actions:
            ax.text(0.5, 0.5, 'No actions recorded', ha='center', va='center')
            return
        actions = [a.replace('_', ' ').replace('train ', '').replace('build ', '').title()
                   for a in actions]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(actions)))
        ax.barh(actions, counts, color=colors, alpha=0.8)
        ax.set_xlabel('Count', fontsize=9, fontweight='bold')
        ax.set_title('Top Meaningful Actions (Wins)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(labelsize=7)
        for i, (action, count) in enumerate(zip(actions, counts)):
            ax.text(count, i, f' {count}', va='center', fontsize=7, fontweight='bold')
    
    def plot_building_composition(self, ax):
        wins = [e for e in self.episodes if e['outcome'] == 'win']
        if not wins:
            ax.text(0.5, 0.5, 'No wins yet', ha='center', va='center')
            return
        building_order = [
            'command_centers', 'orbital_commands', 'planetary_fortresses',
            'supply_depots', 'refineries', 'barracks', 
            'barracks_techlabs', 'barracks_reactors',
            'factories', 'factory_techlabs', 'factory_reactors',
            'starports', 'starport_techlabs', 'starport_reactors',
            'engineering_bays', 'armories', 'fusion_cores',
            'missile_turrets', 'sensor_towers', 'bunkers', 'ghost_academies'
        ]
        averages = []
        labels = []
        for building in building_order:
            values = []
            for e in wins:
                if 'peak_buildings' in e:
                    val = e['peak_buildings'].get(building, 0)
                elif 'final_buildings' in e:
                    val = e['final_buildings'].get(building, 0)
                else:
                    val = e.get(building, 0)
                values.append(val)
            avg = statistics.mean(values) if values else 0
            if avg > 0.3:
                averages.append(avg)
                label = (building.replace('_', ' ')
                        .replace('command centers', 'CCs')
                        .replace('orbital commands', 'Orbitals')
                        .replace('planetary fortresses', 'Planets')
                        .replace('supply depots', 'Depots')
                        .replace('barracks', 'Rax')
                        .replace('factories', 'Fact')
                        .replace('starports', 'Star')
                        .replace('techlabs', 'TL')
                        .replace('reactors', 'Rx')
                        .replace('engineering bays', 'EBay')
                        .replace('missile turrets', 'Turrets')
                        .replace('sensor towers', 'Sensors')
                        .replace('ghost academies', 'Ghost A'))
                labels.append(label.title())
        if not averages:
            ax.text(0.5, 0.5, 'No building data', ha='center', va='center')
            return
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(averages)))
        bars = ax.bar(labels, averages, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax.set_ylabel('Average Peak Count', fontsize=10, fontweight='bold')
        ax.set_title(f'Peak Buildings (n={len(wins)} wins)', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_step_distribution(self, ax):
        steps_by_outcome = defaultdict(list)
        for e in self.episodes:
            steps_by_outcome[e['outcome']].append(e['total_steps'])
        colors = {'win': '#2ecc71', 'loss': '#e74c3c', 'draw': '#95a5a6'}
        for outcome, steps in steps_by_outcome.items():
            if steps:
                ax.hist(steps, bins=30, alpha=0.6, label=f'{outcome.title()} (n={len(steps)})',
                       color=colors.get(outcome, '#3498db'))
        ax.set_xlabel('Steps', fontsize=9, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=9, fontweight='bold')
        ax.set_title('Episode Length Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_winrate_by_corner(self, ax):
        corner_data = defaultdict(lambda: {'wins': 0, 'total': 0})
        for e in self.episodes:
            corner = e.get('start_corner', 'unknown')
            corner_data[corner]['total'] += 1
            if e['outcome'] == 'win':
                corner_data[corner]['wins'] += 1
        corners = list(corner_data.keys())
        winrates = [(corner_data[c]['wins'] / corner_data[c]['total'] * 100) 
                   if corner_data[c]['total'] > 0 else 0 for c in corners]
        totals = [corner_data[c]['total'] for c in corners]
        colors = plt.cm.coolwarm(np.array(winrates) / 100)
        bars = ax.bar(corners, winrates, color=colors, alpha=0.8)
        ax.set_ylabel('Winrate (%)', fontsize=9, fontweight='bold')
        ax.set_title('Winrate by Corner', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.3)
        for bar, total in zip(bars, totals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={total}', ha='center', va='bottom', fontsize=7)
    
    def plot_reward_over_time(self, ax):
        if not self.episodes:
            return
        window = 50
        rewards = []
        episode_nums = []
        for i in range(window, len(self.episodes) + 1):
            chunk = self.episodes[i-window:i]
            avg_reward = statistics.mean(e['total_reward'] for e in chunk)
            rewards.append(avg_reward)
            episode_nums.append(i)
        ax.plot(episode_nums, rewards, linewidth=2, color='#9b59b6')
        ax.fill_between(episode_nums, rewards, alpha=0.3, color='#9b59b6')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Episode', fontsize=9, fontweight='bold')
        ax.set_ylabel('Avg Reward', fontsize=9, fontweight='bold')
        ax.set_title(f'Rolling Reward (window={window})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def create_dashboard(self, save_path='training_dashboard.png'):
        if not self.episodes:
            print("No data to visualize")
            return
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('StarCraft II Bot Training Analysis Dashboard',
                     fontsize=22, fontweight='bold', y=0.98)
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35,
                     left=0.05, right=0.95, top=0.95, bottom=0.05)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[1, :2])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[1, 3])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])
        ax10 = fig.add_subplot(gs[2, 3])
        ax11 = fig.add_subplot(gs[3, :])
        ax11.axis('off')
        self.plot_winrate_over_time(ax1)
        self.plot_game_length_comparison(ax2)
        self.plot_winrate_by_corner(ax3)
        self.plot_economy_progression(ax4)
        self.plot_reward_over_time(ax5)
        self.plot_step_distribution(ax6)
        self.plot_unit_comparison(ax7)
        self.plot_building_composition(ax8)
        self.plot_building_heatmap_by_tier(ax9)
        self.plot_action_distribution(ax10)
        wins = sum(1 for e in self.episodes if e['outcome'] == 'win')
        losses = sum(1 for e in self.episodes if e['outcome'] == 'loss')
        draws = sum(1 for e in self.episodes if e['outcome'] == 'draw')
        total = len(self.episodes)
        winrate = (wins / total * 100) if total > 0 else 0
        if wins > 0:
            avg_win_time = statistics.mean(e['game_time_seconds'] for e in self.episodes if e['outcome'] == 'win')
            avg_win_steps = statistics.mean(e['total_steps'] for e in self.episodes if e['outcome'] == 'win')
        else:
            avg_win_time = 0
            avg_win_steps = 0
        stats_text = f'''
        TRAINING SUMMARY
        ═══════════════════════════════════════════════════════════════════════════════
        Total Episodes: {total}  |  Wins: {wins}  |  Losses: {losses}  |  Draws: {draws}  |  Overall Winrate: {winrate:.1f}%
        Avg Win Time: {avg_win_time:.1f}s ({avg_win_time/60:.1f}min)  |  Avg Win Steps: {avg_win_steps:.0f}
        '''
        ax11.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12, fontweight='bold',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=1))

        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Dashboard saved to: {save_path}")
        plt.close('all')  # Close all figures to free memory

if __name__ == "__main__":
    analyzer = TrainingAnalyzer('training_metrics_worker_*.jsonl')
    analyzer.create_dashboard()
    
    def plot_action_distribution(self, ax):
        wins = [e for e in self.episodes if e['outcome'] == 'win']
        if not wins:
            return
        action_counts = defaultdict(int)
        for w in wins:
            for action, count in w.get('meaningful_actions', {}).items():
                action_counts[action] += count
        top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:12]
        actions, counts = zip(*top_actions) if top_actions else ([], [])
        if not actions:
            ax.text(0.5, 0.5, 'No actions recorded', ha='center', va='center')
            return
        actions = [a.replace('_', ' ').replace('train ', '').replace('build ', '').title()
                   for a in actions]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(actions)))
        ax.barh(actions, counts, color=colors, alpha=0.8)
        ax.set_xlabel('Count', fontsize=9, fontweight='bold')
        ax.set_title('Top Meaningful Actions (Wins)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(labelsize=7)
        for i, (action, count) in enumerate(zip(actions, counts)):
            ax.text(count, i, f' {count}', va='center', fontsize=7, fontweight='bold')
    
    def plot_building_composition(self, ax):
        wins = [e for e in self.episodes if e['outcome'] == 'win']
        if not wins:
            ax.text(0.5, 0.5, 'No wins yet', ha='center', va='center')
            return
        building_order = [
            'command_centers', 'orbital_commands', 'planetary_fortresses',
            'supply_depots', 'refineries', 'barracks', 
            'barracks_techlabs', 'barracks_reactors',
            'factories', 'factory_techlabs', 'factory_reactors',
            'starports', 'starport_techlabs', 'starport_reactors',
            'engineering_bays', 'armories', 'fusion_cores',
            'missile_turrets', 'sensor_towers', 'bunkers', 'ghost_academies'
        ]
        averages = []
        labels = []
        for building in building_order:
            values = []
            for e in wins:
                if 'peak_buildings' in e:
                    val = e['peak_buildings'].get(building, 0)
                elif 'final_buildings' in e:
                    val = e['final_buildings'].get(building, 0)
                else:
                    val = e.get(building, 0)
                values.append(val)
            avg = statistics.mean(values) if values else 0
            if avg > 0.3:
                averages.append(avg)
                label = (building.replace('_', ' ')
                        .replace('command centers', 'CCs')
                        .replace('orbital commands', 'Orbitals')
                        .replace('planetary fortresses', 'Planets')
                        .replace('supply depots', 'Depots')
                        .replace('barracks', 'Rax')
                        .replace('factories', 'Fact')
                        .replace('starports', 'Star')
                        .replace('techlabs', 'TL')
                        .replace('reactors', 'Rx')
                        .replace('engineering bays', 'EBay')
                        .replace('missile turrets', 'Turrets')
                        .replace('sensor towers', 'Sensors')
                        .replace('ghost academies', 'Ghost A'))
                labels.append(label.title())
        if not averages:
            ax.text(0.5, 0.5, 'No building data', ha='center', va='center')
            return
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(averages)))
        bars = ax.bar(labels, averages, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax.set_ylabel('Average Peak Count', fontsize=10, fontweight='bold')
        ax.set_title(f'Peak Buildings (n={len(wins)} wins)', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_step_distribution(self, ax):
        steps_by_outcome = defaultdict(list)
        for e in self.episodes:
            steps_by_outcome[e['outcome']].append(e['total_steps'])
        colors = {'win': '#2ecc71', 'loss': '#e74c3c', 'draw': '#95a5a6'}
        for outcome, steps in steps_by_outcome.items():
            if steps:
                ax.hist(steps, bins=30, alpha=0.6, label=f'{outcome.title()} (n={len(steps)})',
                       color=colors.get(outcome, '#3498db'))
        ax.set_xlabel('Steps', fontsize=9, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=9, fontweight='bold')
        ax.set_title('Episode Length Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_winrate_by_corner(self, ax):
        corner_data = defaultdict(lambda: {'wins': 0, 'total': 0})
        for e in self.episodes:
            corner = e.get('start_corner', 'unknown')
            corner_data[corner]['total'] += 1
            if e['outcome'] == 'win':
                corner_data[corner]['wins'] += 1
        corners = list(corner_data.keys())
        winrates = [(corner_data[c]['wins'] / corner_data[c]['total'] * 100) 
                   if corner_data[c]['total'] > 0 else 0 for c in corners]
        totals = [corner_data[c]['total'] for c in corners]
        colors = plt.cm.coolwarm(np.array(winrates) / 100)
        bars = ax.bar(corners, winrates, color=colors, alpha=0.8)
        ax.set_ylabel('Winrate (%)', fontsize=9, fontweight='bold')
        ax.set_title('Winrate by Corner', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.3)
        for bar, total in zip(bars, totals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={total}', ha='center', va='bottom', fontsize=7)
    
    def plot_reward_over_time(self, ax):
        if not self.episodes:
            return
        window = 50
        rewards = []
        episode_nums = []
        for i in range(window, len(self.episodes) + 1):
            chunk = self.episodes[i-window:i]
            avg_reward = statistics.mean(e['total_reward'] for e in chunk)
            rewards.append(avg_reward)
            episode_nums.append(i)
        ax.plot(episode_nums, rewards, linewidth=2, color='#9b59b6')
        ax.fill_between(episode_nums, rewards, alpha=0.3, color='#9b59b6')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Episode', fontsize=9, fontweight='bold')
        ax.set_ylabel('Avg Reward', fontsize=9, fontweight='bold')
        ax.set_title(f'Rolling Reward (window={window})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def create_dashboard(self, save_path='training_dashboard.png'):
        if not self.episodes:
            print("No data to visualize")
            return
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('StarCraft II Bot Training Analysis Dashboard',
                     fontsize=22, fontweight='bold', y=0.98)
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35,
                     left=0.05, right=0.95, top=0.95, bottom=0.05)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[1, :2])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[1, 3])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])
        ax10 = fig.add_subplot(gs[2, 3])
        ax11 = fig.add_subplot(gs[3, :])
        ax11.axis('off')
        self.plot_winrate_over_time(ax1)
        self.plot_game_length_comparison(ax2)
        self.plot_winrate_by_corner(ax3)
        self.plot_economy_progression(ax4)
        self.plot_reward_over_time(ax5)
        self.plot_step_distribution(ax6)
        self.plot_unit_comparison(ax7)
        self.plot_building_composition(ax8)
        self.plot_building_heatmap_by_tier(ax9)
        self.plot_action_distribution(ax10)
        wins = sum(1 for e in self.episodes if e['outcome'] == 'win')
        losses = sum(1 for e in self.episodes if e['outcome'] == 'loss')
        draws = sum(1 for e in self.episodes if e['outcome'] == 'draw')
        total = len(self.episodes)
        winrate = (wins / total * 100) if total > 0 else 0
        if wins > 0:
            avg_win_time = statistics.mean(e['game_time_seconds'] for e in self.episodes if e['outcome'] == 'win')
            avg_win_steps = statistics.mean(e['total_steps'] for e in self.episodes if e['outcome'] == 'win')
        else:
            avg_win_time = 0
            avg_win_steps = 0
        stats_text = f'''
        TRAINING SUMMARY
        ═══════════════════════════════════════════════════════════════════════════════
        Total Episodes: {total}  |  Wins: {wins}  |  Losses: {losses}  |  Draws: {draws}  |  Overall Winrate: {winrate:.1f}%
        Avg Win Time: {avg_win_time:.1f}s ({avg_win_time/60:.1f}min)  |  Avg Win Steps: {avg_win_steps:.0f}
        '''
        ax11.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12, fontweight='bold',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=1))

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Dashboard saved to: {save_path}")
        plt.close(fig)
        import gc
        gc.collect()

if __name__ == "__main__":
    analyzer = TrainingAnalyzer('training_metrics_worker_*.jsonl')
    analyzer.create_dashboard()