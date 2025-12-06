import json
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

class TrainingAnalyzer:
    """
    Analyze training metrics from the logger with beautiful visualizations.
    """
    def __init__(self, log_file='training_metrics.jsonl'):
        self.log_file = log_file
        self.episodes = []
        self._load_data()
    
    def _load_data(self):
        """Load all episodes from log file"""
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    self.episodes.append(json.loads(line))
            print(f"Loaded {len(self.episodes)} episodes")
        except FileNotFoundError:
            print(f"No log file found at {self.log_file}")
    
    def plot_winrate_over_time(self, ax, window=50):
        """Plot rolling winrate"""
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
        
        ax.plot(episode_nums, winrates, linewidth=2, color='#2ecc71')
        ax.fill_between(episode_nums, winrates, alpha=0.3, color='#2ecc71')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Winrate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Rolling Winrate (window={window})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 100)
    
    def plot_game_length_comparison(self, ax):
        """Compare game lengths between wins and losses"""
        wins = [e for e in self.episodes if e['outcome'] == 'win']
        losses = [e for e in self.episodes if e['outcome'] == 'loss']
        
        if not wins or not losses:
            return
        
        win_times = [e['game_time_seconds'] for e in wins]
        loss_times = [e['game_time_seconds'] for e in losses]
        
        positions = [1, 2]
        data = [win_times, loss_times]
        colors = ['#2ecc71', '#e74c3c']
        
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        showmeans=True, meanline=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['Wins', 'Losses'], fontsize=11, fontweight='bold')
        ax.set_ylabel('Game Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Game Duration Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_unit_comparison(self, ax):
        """Compare final unit counts between wins and losses"""
        wins = [e for e in self.episodes if e['outcome'] == 'win']
        losses = [e for e in self.episodes if e['outcome'] == 'loss']
        
        if not wins or not losses:
            return
        
        unit_types = ['scvs', 'marines', 'marauders', 'siege_tanks', 'thors', 
                     'vikings', 'battlecruisers']
        
        win_avgs = []
        loss_avgs = []
        labels = []
        
        for unit in unit_types:
            win_avg = statistics.mean(e['final_units'].get(unit, 0) for e in wins)
            loss_avg = statistics.mean(e['final_units'].get(unit, 0) for e in losses)
            if win_avg > 0.5 or loss_avg > 0.5:
                win_avgs.append(win_avg)
                loss_avgs.append(loss_avg)
                labels.append(unit.replace('_', ' ').title())
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, win_avgs, width, label='Wins', color='#2ecc71', alpha=0.8)
        ax.bar(x + width/2, loss_avgs, width, label='Losses', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Unit Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Count', fontsize=12, fontweight='bold')
        ax.set_title('Final Unit Composition', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_building_heatmap(self, ax):
        """Show building placement heatmap"""
        wins = [e for e in self.episodes if e['outcome'] == 'win' and 'placement_heatmap' in e]
        
        if not wins:
            ax.text(0.5, 0.5, 'No placement data', ha='center', va='center')
            return
        
        total_heatmap = np.zeros((10, 10))
        for w in wins:
            heatmap = np.array(w['placement_heatmap'])
            total_heatmap += heatmap
        
        im = ax.imshow(total_heatmap, cmap='YlOrRd', interpolation='nearest')
        ax.set_title('Building Placement Heatmap (Wins)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (normalized)', fontsize=11)
        ax.set_ylabel('Y Position (normalized)', fontsize=11)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Placement Density', rotation=270, labelpad=20, fontweight='bold')
        
        # Grid
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels([f'{i/10:.1f}' for i in range(10)])
        ax.set_yticklabels([f'{i/10:.1f}' for i in range(10)])
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    def plot_economy_progression(self, ax):
        """Show economy development over time"""
        wins = [e for e in self.episodes if e['outcome'] == 'win' and e.get('economy_snapshots')]
        
        if not wins:
            return
        
        step_data = defaultdict(lambda: {'minerals': [], 'scvs': [], 'army': []})
        
        for w in wins:
            for snap in w['economy_snapshots']:
                step = snap['step']
                step_data[step]['minerals'].append(snap['minerals'])
                step_data[step]['scvs'].append(snap['scvs'])
                step_data[step]['army'].append(snap['total_army'])
        
        steps = sorted(step_data.keys())
        minerals = [statistics.mean(step_data[s]['minerals']) for s in steps]
        scvs = [statistics.mean(step_data[s]['scvs']) for s in steps]
        army = [statistics.mean(step_data[s]['army']) for s in steps]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(steps, minerals, label='Minerals', color='#f39c12', linewidth=2)
        line2 = ax2.plot(steps, scvs, label='SCVs', color='#3498db', linewidth=2)
        line3 = ax2.plot(steps, army, label='Army Units', color='#e74c3c', linewidth=2)
        
        ax.set_xlabel('Game Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Minerals', fontsize=12, fontweight='bold', color='#f39c12')
        ax2.set_ylabel('Unit Count', fontsize=12, fontweight='bold', color='#3498db')
        ax.set_title('Economy & Army Progression (Wins)', fontsize=14, fontweight='bold')
        
        ax.tick_params(axis='y', labelcolor='#f39c12')
        ax2.tick_params(axis='y', labelcolor='#3498db')
        
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def plot_action_distribution(self, ax):
        """Show most common actions in winning games"""
        wins = [e for e in self.episodes if e['outcome'] == 'win']
        
        if not wins:
            return
        
        action_counts = defaultdict(int)
        for w in wins:
            for action, count in w['action_distribution'].items():
                action_counts[action] += count
        
        # Top 10 actions
        top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        actions, counts = zip(*top_actions)
        
        # Clean up action names
        actions = [a.replace('_', ' ').replace('train ', '').replace('build ', '').title() 
                  for a in actions]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(actions)))
        ax.barh(actions, counts, color=colors, alpha=0.8)
        ax.set_xlabel('Total Count', fontsize=12, fontweight='bold')
        ax.set_title('Most Common Actions (Wins)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add count labels
        for i, (action, count) in enumerate(zip(actions, counts)):
            ax.text(count, i, f' {count}', va='center', fontweight='bold')
    
    def create_dashboard(self, save_path='training_dashboard.png'):
        """Create comprehensive training dashboard"""
        if not self.episodes:
            print("No data to visualize")
            return
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('StarCraft II Bot Training Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Top row
        ax1 = fig.add_subplot(gs[0, :])  # Winrate spans full width
        
        # Middle row
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 2])
        
        # Bottom row
        ax5 = fig.add_subplot(gs[2, :2])  # Economy spans 2 columns
        ax6 = fig.add_subplot(gs[2, 2])
        
        # Generate all plots
        self.plot_winrate_over_time(ax1)
        self.plot_game_length_comparison(ax2)
        self.plot_unit_comparison(ax3)
        self.plot_building_heatmap(ax4)
        self.plot_economy_progression(ax5)
        self.plot_action_distribution(ax6)
        
        # Add stats text box
        wins = sum(1 for e in self.episodes if e['outcome'] == 'win')
        total = len(self.episodes)
        winrate = (wins / total * 100) if total > 0 else 0
        
        stats_text = f'Total Episodes: {total}\nTotal Wins: {wins}\nOverall Winrate: {winrate:.1f}%'
        fig.text(0.02, 0.02, stats_text, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nDashboard saved to: {save_path}")
        plt.show()

if __name__ == "__main__":
    analyzer = TrainingAnalyzer('training_metrics.jsonl')
    analyzer.create_dashboard()