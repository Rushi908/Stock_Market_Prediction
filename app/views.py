import base64
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from io import BytesIO
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

from app.models import *

matplotlib.use('SVG')
window_size = 90
num_days = 90


def register(request):
    try:
        if request.method == 'POST':
            user = User()
            user.username = str(request.POST.get('username')).strip()
            user.password = str(request.POST.get('password')).strip()
            user.role = "user"
            user.save(force_insert=True)
            message = 'User registration done'
            return render(request, 'app/register.html', {'message': message})
        else:
            return render(request, 'app/register.html')
    except Exception as ex:
        return render(request, 'app/register.html', {'message': ex})


def login(request):
    try:
        if request.method == 'POST':
            username = str(request.POST.get("username")).strip()
            password = str(request.POST.get("password")).strip()
            user = User.objects.get(username=username, password=password)
            role = user.role
            if role == "admin":
                request.session['alogin'] = True
                return redirect(add_company)
            elif role == "user":
                request.session['ulogin'] = True
                request.session['username'] = user.username
                return redirect(search_company)
            message = 'Invalid username or password'
        else:
            request.session['alogin'] = False
            request.session['ulogin'] = False
            return render(request, 'app/login.html')
    except User.DoesNotExist:
        message = 'Invalid username or password'
    except Exception as ex:
        message = ex
    return render(request, 'app/login.html', {'message': message})


def add_company(request):
    try:
        if request.method == 'POST':
            cname = str(request.POST.get('name')).strip()
            ticker = str(request.POST.get('ticker')).strip().upper() + ".NS"
            start = datetime.now() - relativedelta(years=5)
            start = start.strftime("%Y-%m-%d")
            end = datetime.now().strftime("%Y-%m-%d")
            dataset = yf.download(ticker, start=start, end=end)
            if dataset.empty:
                raise Exception("Invalid Ticker")
            company = Company()
            company.name = cname
            company.ticker = ticker
            company.save(force_insert=True)
            message = 'Company added successfully'
        else:
            if 'alogin' in request.session and request.session['alogin']:
                return render(request, 'admin/add_company.html')
            else:
                return redirect(login)
    except Exception as ex:
        message = ex
    return render(request, 'admin/add_company.html', {'message': message})


def delete_company(request):
    if 'alogin' in request.session and request.session['alogin']:
        companies = None
        message = ''
        try:
            if request.method == 'POST':
                cname = str(request.POST.get('cname')).strip()
                company = Company.objects.get(name=cname)
                company.delete()
            companies = Company.objects.all()
        except Exception as ex:
            message = ex
        return render(request, 'admin/delete_company.html', {'message': message, 'companies': companies})
    else:
        return redirect(login)


def upload_dataset(request):
    if 'alogin' in request.session and request.session['alogin']:
        companies = None
        message = ''
        try:
            companies = Company.objects.values_list('name', flat=True)
            if request.method == "POST":
                cname = str(request.POST.get('company')).strip()
                company = Company.objects.get(name=cname)
                start = datetime.now() - relativedelta(years=5)
                start = start.strftime("%Y-%m-%d")
                end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                dataset = yf.download(company.ticker, start=start, end=end)
                dataset.to_csv(f'media/{cname}.csv')
                ds = pd.read_csv(f'media/{cname}.csv')
                episode = 20
                train_set = ds.iloc[:, 4:5].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_train_set = scaler.fit_transform(train_set)
                with open(f'media/{cname}.pkl', 'wb') as handle:
                    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
                X_train = []
                y_train = []
                for i in range(window_size, train_set.shape[0]):
                    X_train.append(scaled_train_set[i - window_size:i, 0])
                    y_train.append(scaled_train_set[i, 0])
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50, return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50, return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50))
                model.add(Dropout(0.2))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=episode, batch_size=32)
                model.save(f'media/{cname}.h5')
                message = "Dataset updated successfully"
        except Exception as ex:
            message = ex
        return render(request, 'admin/upload_dataset.html', {'message': message, 'companies': companies})
    else:
        return redirect(login)


def analysis(request):
    try:
        if 'alogin' in request.session and request.session['alogin']:
            cname = str(request.GET.get('company')).strip()
            company = Company.objects.get(name=cname)
            start = datetime.now() - relativedelta(years=5)
            start = start.strftime("%Y-%m-%d")
            end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            dataset = yf.download(company.ticker, start=start, end=end)
            if dataset.empty:
                raise Exception("Failed to fetch data")
            dataset.to_csv(f'media/{cname}.csv')
            ds = pd.read_csv(f'media/{cname}.csv')
            with open(f'media/{cname}.pkl', 'rb') as handle:
                scaler = pickle.load(handle)
            model = load_model(f'media/{cname}.h5')
            updated_on = ds.iloc[-1:, 0:1].values[0][0].split()[0]
            updated_on = datetime.strptime(updated_on, '%Y-%m-%d')
            updated_on = updated_on.strftime("%d %B, %Y")
            ds_new = ds.iloc[- window_size:, 4:5]
            d = 0
            while d < num_days:
                test_set = ds_new.iloc[- window_size:, 0:1].values
                inputs = scaler.transform(test_set)
                X_test = []
                for i in range(window_size, test_set.shape[0] + 1):
                    X_test.append(inputs[i - window_size:i, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                X_test = X_test[0:1, 0:window_size, 0:1]
                prediction = model.predict(X_test)
                prediction = scaler.inverse_transform(prediction)[0][0]
                if d == 0:
                    nxt = prediction
                ds_new.loc[len(ds_new)] = prediction
                d += 1
            test_set = ds.iloc[:, 4:5].values
            inputs = scaler.transform(test_set)
            X_test = []
            for i in range(window_size, test_set.shape[0]):
                X_test.append(inputs[i - window_size:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            prediction = model.predict(X_test)
            prediction = scaler.inverse_transform(prediction)
            analysis_result = ds_new.iloc[:, 0:1].values
            plt.plot(test_set, color='red', label='Actual Stock')
            plt.plot(prediction, color='green', label='Predicted Stock')
            new = np.arange(X_test.shape[0] + window_size, X_test.shape[0] + window_size + num_days)
            plt.plot(new, analysis_result[-num_days:], color='orange', label='Upcoming Stock')
            plt.title('LSTM Analyser')
            plt.xlabel('Days')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.clf()
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')
            max_earning = 0.0
            stock = analysis_result[-num_days:]
            for i in range(1, len(stock)):
                if float(stock[i]) - float(stock[i - 1]) > 0:
                    max_earning += (float(stock[i]) - float(stock[i - 1]))
            decision = True if round(max_earning, 2) > 0 else False
            return render(request, 'admin/analysis.html',
                          {'company': cname, 'updated_on': updated_on, 'prediction': round(nxt, 2),
                           'graphic': graphic,
                           'decision': decision})
        else:
            return redirect(login)
    except FileNotFoundError:
        message = "Error in loading dataset, Try to upload dataset again."
    except Exception as ex:
        message = ex
    return render(request, 'admin/analysis.html', {'message': message})


def search_company(request):
    if 'ulogin' in request.session and request.session['ulogin']:
        companies = None
        message = ''
        try:
            companies = Company.objects.all()
        except Exception as ex:
            message = ex
        return render(request, 'user/search_company.html', {'message': message, 'companies': companies})
    else:
        return redirect(login)


def prediction(request):
    try:
        if 'ulogin' in request.session and request.session['ulogin']:
            cname = str(request.GET.get('company')).strip()
            company = Company.objects.get(name=cname)
            start = datetime.now() - relativedelta(years=5)
            start = start.strftime("%Y-%m-%d")
            end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            dataset = yf.download(company.ticker, start=start, end=end)
            if dataset.empty:
                raise Exception("Failed to fetch data")
            dataset.to_csv(f'media/{cname}.csv')
            ds = pd.read_csv(f'media/{cname}.csv')
            with open(f'media/{cname}.pkl', 'rb') as handle:
                scaler = pickle.load(handle)
            model = load_model(f'media/{cname}.h5')
            updated_on = f"{(datetime.now() - relativedelta(years=5)).strftime('%d %B, %Y')} - {datetime.now().strftime('%d %B, %Y')}"
            ds_new = ds.iloc[- window_size:, 4:5]
            d = 0
            while d < num_days:
                test_set = ds_new.iloc[- window_size:, 0:1].values
                inputs = scaler.transform(test_set)
                X_test = []
                for i in range(window_size, test_set.shape[0] + 1):
                    X_test.append(inputs[i - window_size:i, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                X_test = X_test[0:1, 0:window_size, 0:1]
                prediction = model.predict(X_test)
                prediction = scaler.inverse_transform(prediction)[0][0]
                if d == 0:
                    nxt = prediction
                ds_new.loc[len(ds_new)] = prediction
                d += 1
            test_set = ds.iloc[:, 4:5].values
            inputs = scaler.transform(test_set)
            X_test = []
            for i in range(window_size, test_set.shape[0]):
                X_test.append(inputs[i - window_size:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            prediction = model.predict(X_test)
            prediction = scaler.inverse_transform(prediction)
            analysis_result = ds_new.iloc[:, 0:1].values
            plt.plot(test_set, color='red', label='Actual Stock')
            plt.plot(prediction, color='green', label='Predicted Stock')
            new = np.arange(X_test.shape[0] + window_size, X_test.shape[0] + window_size + num_days)
            plt.plot(new, analysis_result[-num_days:], color='orange', label='Upcoming Stock')
            plt.title('LSTM Analyser')
            plt.xlabel('Days')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.clf()
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')
            max_earning = 0.0
            stock = analysis_result[-num_days:]
            for i in range(1, len(stock)):
                if float(stock[i]) - float(stock[i - 1]) > 0:
                    max_earning += (float(stock[i]) - float(stock[i - 1]))
            decision = True if round(max_earning, 2) > 0 else False
            return render(request, 'user/prediction.html',
                          {'company': cname, 'updated_on': updated_on, 'prediction': round(nxt, 2), 'graphic': graphic,
                           'decision': decision})
        else:
            return redirect(login)
    except FileNotFoundError:
        message = "Error in loading dataset, Try to upload dataset again."
    except Exception as ex:
        message = ex
    return render(request, 'user/prediction.html', {'message': message})
