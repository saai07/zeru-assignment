import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # Load features
    df = pd.read_csv('wallet_features_scores.csv')
    
    # Filter wallets
    low_risk = df[df['credit_score'] >= 700]
    high_risk = df[df['credit_score'] < 400]
    
    # 1. Score distribution plot
    plt.figure(figsize=(12, 6))
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    labels = ['0-100', '100-200', '200-300', '300-400', '400-500', 
              '500-600', '600-700', '700-800', '800-900', '900-1000']
    df['score_range'] = pd.cut(df['credit_score'], bins=bins, labels=labels, right=False)
    
    ax = sns.countplot(x='score_range', data=df, order=labels, palette='viridis')
    plt.title('Wallet Credit Score Distribution', fontsize=16)
    plt.xlabel('Credit Score Range', fontsize=12)
    plt.ylabel('Number of Wallets', fontsize=12)
    plt.xticks(rotation=45)
    
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                f'{height/total:.1%}', ha='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig('score_distribution.png', dpi=300)
    plt.show()
    
    # Behavioral comparison bar plot
    comparison = pd.DataFrame({
        'Feature': ['Repayment Ratio', 'Utilization', 'TX Frequency'],
        'Low Risk (700-1000)': [
            low_risk['repay_ratio'].mean(),
            #low_risk['liquidation_ratio'].mean(),
            low_risk['utilization'].mean(),
            low_risk['tx_frequency'].mean(),
        ],
        'High Risk (0-400)': [
            high_risk['repay_ratio'].mean(),
            #high_risk['liquidation_ratio'].mean(),
            high_risk['utilization'].mean(),
            high_risk['tx_frequency'].mean(),
        ]
    })
    
    plot_df = comparison.melt(id_vars='Feature', 
                            var_name='Risk Category', 
                            value_name='Value')
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Feature', y='Value', hue='Risk Category', data=plot_df, palette='RdYlGn')
    
    plt.title('Wallet Behavior Comparison: Low vs High Risk', fontsize=16)
    plt.xlabel('Behavioral Feature', fontsize=12)
    plt.ylabel('Average Value', fontsize=12)
    plt.xticks(rotation=15)
    plt.legend(title='Risk Category')
    plt.tight_layout()
    
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 10), 
                   textcoords='offset points')
    
    plt.savefig('risk_comparison.png', dpi=300)
    plt.show()
    
    print("\nKey Behavioral Differences:")
    print(f"- Repayment Ratio: Low risk wallets repay {low_risk['repay_ratio'].mean():.0%} of debts vs "
          f"high risk: {high_risk['repay_ratio'].mean():.0%}")
    print(f"- Liquidation Ratio: Low risk wallets have {low_risk['liquidation_ratio'].mean():.0%} liquidations vs "
          f"high risk: {high_risk['liquidation_ratio'].mean():.0%}")
    print(f"- Utilization: Low risk use {low_risk['utilization'].mean():.0%} of deposits vs "
          f"high risk: {high_risk['utilization'].mean():.0%}")
    print(f"- TX Frequency: Low risk: {low_risk['tx_frequency'].mean():.1f}/day vs "
          f"high risk: {high_risk['tx_frequency'].mean():.1f}/day")

if __name__ == "__main__":
    main()