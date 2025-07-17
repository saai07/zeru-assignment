import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def extract_features(transactions):
    records = []
    for tx in transactions:
        try:
            amount = float(tx['actionData']['amount'])
            asset_price = float(tx['actionData']['assetPriceUSD'])
            usd_value = amount * asset_price
            
            records.append({
                'user': tx['userWallet'],
                'type': tx['action'],
                'timestamp': tx['timestamp'],
                'amount_usd': usd_value
            })
        except (KeyError, TypeError, ValueError):
            continue
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    features = []
    wallets = df['user'].unique()
    
    for wallet in wallets:
        wallet_txs = df[df['user'] == wallet]
        wallet_txs = wallet_txs.sort_values('timestamp')
        type_counts = wallet_txs['type'].value_counts()
        
        n_deposit = type_counts.get('deposit', 0)
        n_borrow = type_counts.get('borrow', 0)
        n_repay = type_counts.get('repay', 0)
        n_redeem = type_counts.get('redeemunderlying', 0)
        n_liquidation = type_counts.get('liquidationcall', 0)
        n_total = len(wallet_txs)
        
        liquidated = int(n_liquidation > 0)
        
        repay_ratio = n_repay / n_borrow if n_borrow > 0 else 1.0
        liquidation_ratio = n_liquidation / n_borrow if n_borrow > 0 else 0.0
        
        first_ts = wallet_txs['timestamp'].min()
        last_ts = wallet_txs['timestamp'].max()
        wallet_age = max(last_ts - first_ts, 0) / 86400
        tx_frequency = n_total / wallet_age if wallet_age > 0 else n_total
        
        if n_total > 1:
            time_diffs = np.diff(wallet_txs['timestamp'].sort_values())
            time_var = np.var(time_diffs) if len(time_diffs) > 0 else 0
        else:
            time_var = 0
            
        deposit_amount = wallet_txs[wallet_txs['type'] == 'deposit']['amount_usd'].sum()
        borrow_amount = wallet_txs[wallet_txs['type'] == 'borrow']['amount_usd'].sum()
        redeem_amount = wallet_txs[wallet_txs['type'] == 'redeemunderlying']['amount_usd'].sum()
        repay_amount = wallet_txs[wallet_txs['type'] == 'repay']['amount_usd'].sum()
        
        net_deposit = max(deposit_amount - redeem_amount, 0)
        net_borrow = max(borrow_amount - repay_amount, 0)
        utilization = net_borrow / net_deposit if net_deposit > 0 else 0
        avg_tx_size = wallet_txs['amount_usd'].mean()
        
        features.append({
            'user': wallet,
            'n_deposit': n_deposit,
            'n_borrow': n_borrow,
            'n_repay': n_repay,
            'n_redeem': n_redeem,
            'n_liquidation': n_liquidation,
            'n_total': n_total,
            'liquidated': liquidated,
            'repay_ratio': repay_ratio,
            'liquidation_ratio': liquidation_ratio,
            'wallet_age': wallet_age,
            'tx_frequency': tx_frequency,
            'time_variance': time_var,
            'net_deposit': net_deposit,
            'net_borrow': net_borrow,
            'utilization': utilization,
            'avg_tx_size': avg_tx_size
        })
    
    return pd.DataFrame(features)

def compute_credit_scores(input_file, output_json, output_csv):
    # Load and parse JSON
    with open(input_file, 'r') as f:
        data = f.read()
        transactions = json.loads(data)
    
    # Extract features
    feature_df = extract_features(transactions)
    
    if feature_df.empty:
        # Create empty outputs
        with open(output_json, 'w') as f:
            json.dump({'wallets': []}, f)
        feature_df.to_csv(output_csv, index=False)
        return
    
    # Replace inf and large values
    feature_df = feature_df.replace([np.inf, -np.inf], 1e10)
    
    # Model setup
    X = feature_df.drop(columns=['user'])
    model = IsolationForest(
        n_estimators=1000,
        max_samples=min(256, len(X)),
        contamination=0.05,
        random_state=42,
        verbose=0
    )
    model.fit(X)
    
    anomaly_scores = model.decision_function(X)
    anomaly_scores = -anomaly_scores  # Invert: higher = better
    
    scaler = MinMaxScaler(feature_range=(0, 1000))
    credit_scores = scaler.fit_transform(anomaly_scores.reshape(-1, 1)).flatten()
    
    # Assign scores to feature DF
    feature_df['credit_score'] = credit_scores
    
    # Save to CSV (features + scores)
    feature_df.to_csv(output_csv, index=False)
    
    # Prepare JSON output (only user and credit_score)
    json_output = []
    for _, row in feature_df.iterrows():
        json_output.append({
            'user': row['user'],
            'credit_score': round(row['credit_score'], 2)
        })
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump({'wallets': json_output}, f, indent=2)

if __name__ == "__main__":
    compute_credit_scores(
        input_file='input_transactions.json',
        output_json='output_scores.json',
        output_csv='wallet_features_scores.csv'
    )
    print("Done")