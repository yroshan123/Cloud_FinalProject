from flask import Flask, request, render_template, redirect, session, url_for
import os
import re
import mysql.connector
import pandas as pd
from mysql.connector import errorcode
from flask_session import Session
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from flask import Flask, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os



app = Flask(__name__)
app.secret_key = '@dkjgfjgfhkj jxbjljv kjxgvljklkj'
dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(dir_path, 'static', 'files')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

config = {
    'host': 'krogerdatabase.mysql.database.azure.com',
    'user': 'ccfinal',
    'password': 'Ccfinal123',
    'database': 'cckroger',
    'port':3306,
    'ssl_ca': os.path.join(dir_path,'ssl','BaltimoreCyberTrustRoot.crt.pem'),

    'connect_timeout': 50000
}




def get_data_from_db():
    conn = mysql.connector.connect(**config)
    cur = conn.cursor()
    
    query = """
    SELECT h.HSHD_NUM, t.BASKET_NUM, t.PURCHASE_, p.PRODUCT_NUM, p.DEPARTMENT, p.COMMODITY,
           t.SPEND, t.UNITS, t.STORE_R, t.WEEK_NUM, t.YEAR, h.L, h.AGE_RANGE, h.MARITAL, 
           h.INCOME_RANGE, h.HOMEOWNER, h.HSHD_COMPOSITION, h.HH_SIZE, h.CHILDREN
    FROM households AS h
    JOIN transactions AS t ON h.HSHD_NUM = t.HSHD_NUM
    JOIN products AS p ON t.PRODUCT_NUM = p.PRODUCT_NUM
    """
    
    cur.execute(query)
    data = cur.fetchall()
    conn.close()
    
    return data


def get_https_url(item,data):
    return url_for(item,username=data,_external=True, _scheme='http')

@app.route('/',methods=['GET','POST'])
def homepage():
    msg=''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username'] 
        password = request.form['password']
        conn = mysql.connector.connect(**config)
        print("connection established")
        cur = conn.cursor()
        print(username,password)
        cur.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password, ))
        user = cur.fetchone()
        if user: 
            session['loggedin'] = True
            session['username'] = username
            return redirect(get_https_url('profile',username))
        else:
            # Account doesnt exist
            msg = 'Incorrect username/password!'
    return render_template("homepage.html",msg=msg)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # 🔗 Connect to MySQL
        conn = mysql.connector.connect(**config)
        cur = conn.cursor()

        # 🛒 Fetch transactions table
        cur.execute("SELECT * FROM transactions")
        transactions_data = cur.fetchall()
        transactions_columns = [desc[0] for desc in cur.description]
        transactions = pd.DataFrame(transactions_data, columns=transactions_columns)

        # 🛒 Fetch products table
        cur.execute("SELECT * FROM products")
        products_data = cur.fetchall()
        products_columns = [desc[0] for desc in cur.description]
        products = pd.DataFrame(products_data, columns=products_columns)

        cur.close()
        conn.close()

        # 🛠️ Clean column names
        transactions.columns = transactions.columns.str.strip()
        products.columns = products.columns.str.strip()

        if transactions.empty or products.empty:
            return "<h4>⚠️ No transactions or products data found in database!</h4>"

        # 🛒 Merge transactions with products
        merged = transactions.merge(products, on='PRODUCT_NUM', how='left')

        if merged.empty:
            return "<h4>⚠️ Merged data is empty. Check data integrity!</h4>"

        # 🔥 Force SPEND to numeric to avoid type errors
        merged['SPEND'] = pd.to_numeric(merged['SPEND'], errors='coerce')
        merged = merged.dropna(subset=['SPEND'])

        if merged.empty:
            return "<h4>⚠️ Merged data is empty after cleaning SPEND values!</h4>"

        # 📊 Create Basket Data (Pivot Table)
        basket = merged.pivot_table(index='BASKET_NUM', columns='COMMODITY', values='SPEND', aggfunc='sum', fill_value=0)

        if basket.empty:
            return "<h4>⚠️ No basket data available after pivot. Please check source tables.</h4>"

        # 🔥 Binarize basket (purchase: 1 / no-purchase: 0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)

        # 🎯 Auto-pick Top Purchased Product
        product_counts = basket.sum().sort_values(ascending=False)
        
        if product_counts.empty:
            return "<h4>⚠️ No products found to predict.</h4>"

        target_product = product_counts.index[0]

        if target_product not in basket.columns:
            return "<h4>⚠️ Target product not found in basket data!</h4>"

        # 🧹 Prepare data for ML
        X = basket.drop(columns=[target_product])
        y = basket[target_product]

        if len(y.unique()) < 2:
            return "<h4>⚠️ Not enough variety in target labels to train model.</h4>"

        # 📚 Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 🌳 Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 📋 Predict & Evaluate
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # 📈 Feature Importance
        feature_importance = pd.DataFrame({
            'Product': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # 📊 Save Feature Importance Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_importance['Product'][:10][::-1], feature_importance['Importance'][:10][::-1], color='teal')
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top Features Influencing Purchase of {target_product}')
        plt.tight_layout()

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'feature_importance.png')
        plt.savefig(img_path)
        plt.close()

        # 🚀 Send results to template
        return render_template('predict.html',
                               target_product=target_product,
                               metrics=pd.DataFrame(report).T,
                               cm=cm,
                               feature_plot_path='static/files/feature_importance.png')

    except Exception as e:
        return f"<h4>⚠️ An error occurred while processing basket prediction:<br><br>{str(e)}</h4>"

       

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction failed!"





@app.route('/logout')
def logout():
   session.pop('username', None)
   return render_template("homepage.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        conn = mysql.connector.connect(**config)
        cur = conn.cursor()
        print(username, password, email)
        cur.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cur.fetchone()
        if user:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            cur.execute('INSERT INTO users VALUES (%s, %s,%s)', (username, password, email))
            conn.commit()
            session['loggedin'] = True
            session['username'] = username
            return redirect(get_https_url('profile',username))
        print("hai",msg)
    return render_template("register.html",msg=msg)
            

@app.route('/profile/<string:username>',methods=['GET','POST'])
def profile(username):
    if 'loggedin' in session:
        return render_template('profile.html',username=username) 
    return redirect(get_https_url('homepage'))

@app.route('/Search', methods=['GET','POST'])
def Search():
    msg = ''
    if request.method == 'POST' and 'search' in request.form :
        print("came to search[POST]")
        number = request.form['search']
        if not re.match(r'\d+', number):
             msg = "enter a valid household number"
        else:
            conn = mysql.connector.connect(**config)
            cur = conn.cursor()
            cur.execute("SELECT h.HSHD_NUM, t.BASKET_NUM, t.PURCHASE_, p.PRODUCT_NUM, p.DEPARTMENT, p.COMMODITY,t.SPEND, t.UNITS, t.STORE_R, t.WEEK_NUM, t.YEAR, h.L, h.AGE_RANGE,h.MARITAL,h.INCOME_RANGE, h.HOMEOWNER, h.HSHD_COMPOSITION, h.HH_SIZE, h.CHILDREN FROM households AS h JOIN transactions AS t ON h.HSHD_NUM = t.HSHD_NUM JOIN products AS p ON t.PRODUCT_NUM = p.PRODUCT_NUM where h.HSHD_NUM=%s",(number,))
            data=cur.fetchall()
            if data:
                return render_template('Search.html', data= data)
            else:
                msg="Not Data Found for the input "
                return render_template('Search.html', msg=msg)
        return render_template('Search.html', msg=msg)
    else:
        print("came to search[GET]")
        conn = mysql.connector.connect(**config)
        print("connected to database")
        cur = conn.cursor()
        testquery="SELECT h.HSHD_NUM, t.BASKET_NUM, t.PURCHASE_, p.PRODUCT_NUM, p.DEPARTMENT, p.COMMODITY,t.SPEND, t.UNITS, t.STORE_R, t.WEEK_NUM, t.YEAR, h.L, h.AGE_RANGE,h.MARITAL,h.INCOME_RANGE, h.HOMEOWNER, h.HSHD_COMPOSITION, h.HH_SIZE, h.CHILDREN FROM households AS h JOIN transactions AS t ON h.HSHD_NUM = t.HSHD_NUM JOIN products AS p ON t.PRODUCT_NUM = p.PRODUCT_NUM WHERE h.HSHD_NUM=10 ORDER BY h.HSHD_NUM, t.BASKET_NUM, t.PURCHASE_, p.PRODUCT_NUM, p.DEPARTMENT, p.COMMODITY"
        cur.execute(testquery)
        print("test query executed")
        data=cur.fetchmany(1000)
        print("data fetched")
        return render_template('Search.html', data= data)



@app.route('/dashboard')
def dashboard():
   return render_template("dashboard.html")

@app.route('/upload', methods=['GET','POST'])
def upload():
    msg = ''
    if request.method == 'POST':
        hdata=request.files['households']
        tdata=request.files['transactions']
        pdata=request.files['products']
        conn = mysql.connector.connect(**config)
        cur = conn.cursor()
        if hdata.filename == '' or tdata.filename == '' or pdata.filename == '' :
            msg='No Files passed'
            return render_template('upload.html', msg=msg)
        else:
            #household data
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],hdata.filename)
            hdata.save(file_path)
            col_names=['HSHD_NUM','L','AGE_RANGE','MARITAL','INCOME_RANGE','HOMEOWNER','HSHD_COMPOSITION','HH_SIZE','CHILDREN']
            csvData = pd.read_csv(file_path,names=col_names,header=0)
            query='INSERT INTO households (HSHD_NUM,L,AGE_RANGE,MARITAL,INCOME_RANGE,HOMEOWNER,HSHD_COMPOSITION,HH_SIZE,CHILDREN) VALUES'
            for i,row in csvData.iterrows():
                if(pd.isna(row['CHILDREN'])):
                    query += '{},'.format((row['HSHD_NUM'],row['L'],row['AGE_RANGE'],row['MARITAL'],row['INCOME_RANGE'],row['HOMEOWNER'],row['HSHD_COMPOSITION'],row['HH_SIZE'],'null'))
                else :
                    query += '{},'.format((row['HSHD_NUM'],row['L'],row['AGE_RANGE'],row['MARITAL'],row['INCOME_RANGE'],row['HOMEOWNER'],row['HSHD_COMPOSITION'],row['HH_SIZE'],row['CHILDREN']))
            query = query[:len(query)-1]
            cur.execute(query)


            #transaction data
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],tdata.filename)
            tdata.save(file_path)
            print("file path",file_path)
            col_names=['BASKET_NUM','HSHD_NUM','PURCHASE_','PRODUCT_NUM','SPEND','UNITS','STORE_R','WEEK_NUM','YEAR']
            csvData = pd.read_csv(file_path,names=col_names,header=0,nrows=10000)
            query='INSERT INTO transactions (BASKET_NUM,PURCHASE_,SPEND,UNITS,STORE_R,WEEK_NUM,YEAR,HSHD_NUM,PRODUCT_NUM) VALUES'
            for i,row in csvData.iterrows():
                query += '{},'.format((row['BASKET_NUM'],row['PURCHASE_'],row['SPEND'],row['UNITS'],row['STORE_R'],row['WEEK_NUM'],row['YEAR'],row['HSHD_NUM'],row['PRODUCT_NUM']))
            query = query[:len(query)-1]
            cur.execute(query)


            # Products data
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],pdata.filename)
            pdata.save(file_path)
            col_names=['PRODUCT_NUM','DEPARTMENT','COMMODITY','BRAND_TY','NATURAL_ORGANIC_FLAG']
            csvData = pd.read_csv(file_path,names=col_names,header=0)
            query='INSERT INTO products (PRODUCT_NUM,DEPARTMENT,COMMODITY,BRAND_TYPE,NATURAL_ORGANIC_FLAG) VALUES'
            for i,row in csvData.iterrows():
                query += '{},'.format((row['PRODUCT_NUM'],row['DEPARTMENT'],row['COMMODITY'],row['BRAND_TY'],row['NATURAL_ORGANIC_FLAG']))
            query = query[:len(query)-1]
            cur.execute(query)
            
            conn.commit()
            msg='Sucessfully Inserted data !!!!!'
            print(msg)
            return render_template("upload.html", msg=msg)
    else:
        msg="unable to insert data"
        return render_template("upload.html")
@app.route('/churn')
def churn():
    query = """
    SELECT hshd_num, COUNT(DISTINCT basket_num) AS frequency,
           SUM(spend) AS total_spent,
           MAX(purchase_) AS last_purchase
    FROM transactions
    GROUP BY hshd_num
    HAVING total_spent > 0
    """
    conn = mysql.connector.connect(**config)
    cur = conn.cursor()

    cur.execute(query)
    result = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    df = pd.DataFrame(result, columns=columns)

    cur.close()
    conn.close()

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
    return render_template('churn.html', metrics=pd.DataFrame(report).T, test_size=len(y_test), threshold=int(threshold))


    
if __name__=="__main__":
    app.run(debug=True)
    




