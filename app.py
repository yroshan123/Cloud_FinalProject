from flask import Flask, request, render_template, redirect
from google.cloud import bigquery
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from mlxtend.frequent_patterns import apriori
from werkzeug.utils import secure_filename

app = Flask(__name__)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"
client = bigquery.Client()

@app.route('/')
def index():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect('/home')
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        dataset_id = "retailinsightproject.retaildataset_us"
        files = {
            'households': request.files['households'],
            'transactions': request.files['transactions'],
            'products': request.files['products']
        }
        success_msgs = []
        for name, file in files.items():
            filename = secure_filename(file.filename)
            file.save(filename)
            df = pd.read_csv(filename)
            table_id = f"{dataset_id}.{name}"
            job_config = bigquery.LoadJobConfig(
                autodetect=True,
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=1
            )
            with open(filename, "rb") as source_file:
                job = client.load_table_from_file(source_file, table_id, job_config=job_config)
                job.result()
                success_msgs.append(f"✅ {name} uploaded successfully.")
        return render_template('upload.html', messages=success_msgs)
    return render_template('upload.html')

@app.route('/datapull10')
def datapull10():
    query = """
    SELECT t.hshd_num, t.basket_num, t.purchase_, p.product_num, 
           p.department, p.commodity, t.spend, t.units
    FROM `retailinsightproject.retaildataset_us.transactions` t
    JOIN `retailinsightproject.retaildataset_us.products` p
    ON t.product_num = p.product_num
    WHERE t.hshd_num = 10
    ORDER BY t.hshd_num, t.basket_num, t.purchase_
    """
    results = client.query(query).result()
    rows = [list(row.values()) for row in results]
    headers = ["HSHD_NUM", "BASKET_NUM", "DATE", "PRODUCT_NUM", "DEPARTMENT", "COMMODITY", "SPEND", "UNITS"]
    return render_template('datapull.html', rows=rows, headers=headers)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        hshd = request.form['hshd_num']
        query = f"""
        SELECT t.hshd_num, t.basket_num, t.purchase_, p.product_num, 
               p.department, p.commodity, t.spend, t.units
        FROM `retailinsightproject.retaildataset_us.transactions` t
        JOIN `retailinsightproject.retaildataset_us.products` p
        ON t.product_num = p.product_num
        WHERE t.hshd_num = {hshd}
        ORDER BY t.hshd_num, t.basket_num, t.purchase_
        """
        results = client.query(query).result()
        rows = [list(row.values()) for row in results]
        headers = ["HSHD_NUM", "BASKET_NUM", "DATE", "PRODUCT_NUM", "DEPARTMENT", "COMMODITY", "SPEND", "UNITS"]
        return render_template('search.html', rows=rows, hshd=hshd, headers=headers)
    return render_template('search.html')

@app.route('/basket')
def basket():
    query = """
    SELECT basket_num, product_num
    FROM `retailinsightproject.retaildataset_us.transactions`
    LIMIT 10000
    """
    df = client.query(query).to_dataframe()

    basket_df = df.groupby(['basket_num', 'product_num']).size().unstack(fill_value=0)
    basket_df = basket_df.applymap(lambda x: 1 if x > 0 else 0)
    frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(map(str, x)))
    return render_template('basket.html', itemsets=frequent_itemsets)

@app.route('/churn')
def churn():
    query = """
    SELECT hshd_num, COUNT(DISTINCT basket_num) AS frequency,
           SUM(spend) AS total_spent,
           MAX(purchase_) AS last_purchase
    FROM `retailinsightproject.retaildataset_us.transactions`
    GROUP BY hshd_num
    HAVING total_spent > 0
    """
    df = client.query(query).to_dataframe()

    if df.shape[0] < 10:
        return "<h4>⚠️ Not enough data to train churn model. At least 10 records needed.</h4>"

    df['last_purchase'] = pd.to_datetime(df['last_purchase'])
    max_date = df['last_purchase'].max()
    df['days_since'] = (max_date - df['last_purchase']).dt.days

    threshold = df['days_since'].quantile(0.75)
    df['churn'] = df['days_since'] > threshold

    X = df[['frequency', 'total_spent', 'days_since']]
    y = df['churn']

    if len(y.unique()) < 2:
        return "<h4>⚠️ Not enough variety in churn labels to train model.</h4>"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    return render_template('churn.html', metrics=pd.DataFrame(report).T, test_size=len(y_test), threshold=int(threshold), message=None)

@app.route('/dashboard')
def dashboard():
    query1 = """
    SELECT FORMAT_DATE('%Y-%m', purchase_) AS month, SUM(spend) AS total
    FROM `retailinsightproject.retaildataset_us.transactions`
    GROUP BY month
    ORDER BY month
    """
    line_results = client.query(query1).result()
    line_data = {"labels": [], "values": []}
    for row in line_results:
        line_data["labels"].append(row["month"])
        line_data["values"].append(row["total"])

    query2 = """
    SELECT income_range, COUNT(DISTINCT hshd_num) AS households
    FROM `retailinsightproject.retaildataset_us.households`
    GROUP BY income_range
    ORDER BY households DESC
    LIMIT 5
    """
    income_results = client.query(query2).result()
    income_data = {"labels": [], "values": []}
    for row in income_results:
        income_data["labels"].append(row["income_range"])
        income_data["values"].append(row["households"])

    query3 = """
    SELECT brand_ty, SUM(spend) AS total
    FROM `retailinsightproject.retaildataset_us.products` p
    JOIN `retailinsightproject.retaildataset_us.transactions` t
    ON p.product_num = t.product_num
    GROUP BY brand_ty
    """
    brand_results = client.query(query3).result()
    brand_data = {"labels": [], "values": []}
    for row in brand_results:
        brand_data["labels"].append(row["brand_ty"])
        brand_data["values"].append(row["total"])

    query4 = """
    SELECT EXTRACT(MONTH FROM purchase_) AS month, SUM(spend) AS total
    FROM `retailinsightproject.retaildataset_us.transactions`
    GROUP BY month
    ORDER BY month
    """
    season_results = client.query(query4).result()
    season_data = {"labels": [], "values": []}
    for row in season_results:
        season_data["labels"].append(int(row["month"]))
        season_data["values"].append(row["total"])

    query5 = """
    SELECT commodity, COUNT(*) AS purchases
    FROM `retailinsightproject.retaildataset_us.products` p
    JOIN `retailinsightproject.retaildataset_us.transactions` t
    ON p.product_num = t.product_num
    GROUP BY commodity
    ORDER BY purchases DESC
    LIMIT 5
    """
    basket_results = client.query(query5).result()
    basket_data = {"labels": [], "values": []}
    for row in basket_results:
        basket_data["labels"].append(row["commodity"])
        basket_data["values"].append(row["purchases"])

    return render_template('dashboard.html',
                           line_data=line_data,
                           income_data=income_data,
                           brand_data=brand_data,
                           season_data=season_data,
                           basket_data=basket_data)

if __name__ == '__main__':
    app.run(debug=True, port=8080)