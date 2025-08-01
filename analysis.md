## Wallet Credit Score Analysis
# Executive Summary
This analysis evaluates wallet behavior in the Aave V2 protocol using unsupervised machine learning to assign credit scores (0-1000). Higher scores indicate reliable users, while lower scores signal risky behavior. The Isolation Forest algorithm identified key behavioral patterns that strongly correlate with credit risk.

# Score Distribution
<img width="3600" height="1800" alt="score_distribution" src="https://github.com/user-attachments/assets/843c580b-5f65-466b-986f-d84e1e389e70" />

# Behavior of wallets in the lower range and behavior of wallets in the higher range.

<img width="3600" height="2400" alt="risk_comparison" src="https://github.com/user-attachments/assets/92276e47-4326-4d32-958a-f49d0b0f979e" />

# Key Behavioral Differences:
- Repayment Ratio: Low risk wallets repay 96% of debts vs high risk: 87%
- Liquidation Ratio: Low risk wallets have 5% liquidations vs high risk: 0%
- Utilization: Low risk use 88% of deposits vs high risk: 276323362569%
- TX Frequency: Low risk: 5.1/day vs high risk: 566.6/day

# Scatter plots
<img width="5400" height="3000" alt="scatter_analysis" src="https://github.com/user-attachments/assets/ea1691b9-9490-494b-93ac-4842496cd84e" />

