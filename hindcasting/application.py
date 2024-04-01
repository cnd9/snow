from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import boto3
import pandas as pd
import io
from datetime import datetime, timedelta
from constants import ACCESS_KEY, SECRET_KEY, S3_BUCKET_NAME, OBS_DISPLAY_NAMES, OBS_DISPLAY_COLUMNS
import matplotlib
import json

matplotlib.use('Agg')  # Use this line before importing pyplot
import matplotlib.pyplot as plt
import base64
import io
from users import users

pd.options.mode.chained_assignment = None

s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,  # os.getenv('ACCESS_KEY'),
    aws_secret_access_key=SECRET_KEY  # os.getenv('SECRET_KEY')
)


def load_data_from_s3(file_path):
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_path)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        return df
    except:
        return pd.DataFrame()


def convert_units(df):
    # Celsius to Fahrenheit
    df['air_temp_set_1'] = df['air_temp_set_1'] * 9 / 5 + 32
    # m/s to mph
    df['wind_speed_set_1'] = df['wind_speed_set_1'] * 2.23694
    # mm to inches
    df['precip_accum_one_hour_set_1'] = df['precip_accum_one_hour_set_1'] / 25.4
    return df


application = Flask(__name__)
application.secret_key = 'your_secret_key_here'
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(application)


class User(UserMixin):
    def __init__(self, user_id, dates, name):
        self.id = user_id
        self.dates = dates
        self.name = name


# User Loader
@login_manager.user_loader
def user_loader(user_id):
    if user_id not in users:
        return None
    user_info = users[user_id]
    return User(user_id, user_info['dates'], user_info['name'])


def plot_weather_data(df, date):
    """Generate plots for temperature, wind speed, and accumulated precipitation."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])

    # Temperature plot
    axs[0].plot(df['Date_Time'], df['air_temp_set_1'], label='BIGMT')
    axs[0].axvline(pd.to_datetime(date), color='r', linestyle='--', label='Forecast Date')
    axs[0].set_title('Air Temperature (°F)')
    axs[0].set_ylabel('Temperature (°F)')
    axs[0].legend()

    # Wind Speed plot
    axs[1].plot(df['Date_Time'], df['wind_speed_set_1'], label='BIGMT')
    axs[1].axvline(pd.to_datetime(date), color='r', linestyle='--', label='Forecast Date')
    axs[1].set_title('Wind Speed (mph)')
    axs[1].set_ylabel('Speed (mph)')
    axs[1].legend()

    # Accumulated Precipitation plot
    axs[2].plot(df['Date_Time'], df['precip_accum_one_hour_set_1'], label='BIGMT')
    axs[2].axvline(pd.to_datetime(date), color='r', linestyle='--', label='Forecast Date')
    axs[2].set_title('Precipitation (inches per hour)')
    axs[2].set_ylabel('Precipitation (inches per hour)')
    axs[2].legend()

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf8')


@application.route('/')
def root():
    if current_user.is_authenticated:
        return redirect(url_for('user_home'))
    return redirect(url_for('login'))


@application.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('user_home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            user_obj = User(username, user['dates'], user['name'])
            login_user(user_obj)
            return redirect(url_for('user_home'))
        flash('Invalid username or password')
    return render_template('login.html')


@application.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@application.route('/user_home')
@login_required
def user_home():
    user_dates = users[current_user.id]['dates']
    return render_template('user_home.html', dates=user_dates, user=current_user.name)


def generate_dropdown(row_id):
    return f'''<select name="usefulness_{row_id}" class="usefulness-dropdown">
        <option value="">Not Used</option>
        <option value="1">Slightly Useful</option>
        <option value="2">Moderately Useful</option>
        <option value="3">Very Useful</option>
    </select>'''


def assign_row_class(row, date):
    print(date)
    print(row['avalanche_link_text'])
    if row['startDate'] == date or date in row['avalanche_link_text']:
        return 'highlight-row'
    else:
        return 'normal-row'


# Apply the function to both DataFrames
# Assuming `row_id` is a column in your DataFrame
def generate_table_html(df, columns):
    # Start the table and add the header row
    html = '<table class="table table-hover">'
    html += '<thead><tr>'
    for col in columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead>'
    html += '<tbody>'

    # Iterate over DataFrame rows
    for index, row in df.iterrows():
        # Determine row class based on some condition, e.g., observer type
        row_class = row['row_class']
        html += f'<tr class="{row_class}">'

        # Add each cell in the row
        for col in columns:
            cell_value = row[col]
            # If the cell contains HTML (like links), ensure it's safe to render directly
            if isinstance(cell_value, str) and (cell_value.startswith('<a ') or cell_value.startswith('<div ')):
                cell_html = cell_value
            else:
                cell_html = f'{cell_value}'  # Convert to string and HTML-escape if necessary
            html += f'<td>{cell_html}</td>'

        html += '</tr>'

    # Close the table tags
    html += '</tbody></table>'
    return html


@application.route('/date_landing/<date>')
@login_required
def date_landing(date):
    if date not in users[current_user.id]['dates']:
        flash("You do not have access to this date.")
        return redirect(url_for('user_home'))
    date_object = datetime.strptime(date, '%Y-%m-%d')
    formatted_date = date_object.strftime('%A %Y-%m-%d')
    formatted_date_only = date_object.strftime('%Y-%m-%d')
    start_date = date_object - timedelta(days=7)
    week_later = date_object + timedelta(days=7)
    # Load data from S3
    file_path = 'observations/general/fac_2022_2023.csv'
    df = load_data_from_s3(file_path)
    df['row_class'] = df.apply(assign_row_class, axis=1, args=(formatted_date_only,))
    df['Useful?'] = df['id'].apply(generate_dropdown)
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['snowpack_description'] = df['snowpack_description'].apply(
        lambda x: f'<div class="scrollable-content">{x}</div>')
    df['link'] = df['link'].apply(
        lambda x: f'<a href="{x}" target="_blank">View</a>')

    filtered_df = df[(df['startDate'] > start_date) & (df['startDate'] < date_object)]
    filtered_df = filtered_df.sort_values(by='startDate', ascending=False)
    filtered_df = filtered_df.rename(columns=OBS_DISPLAY_NAMES)
    filtered_df['Date'] = filtered_df['Date'].apply(lambda x: x.strftime('%A %Y-%m-%d'))
    # Separate DataFrames for Forecaster/Professional and Public observations
    forecaster_professional_df = filtered_df[
        filtered_df['observerType'].isin(['professional', 'forecaster'])]  # .sort_values(by='Date', ascending=True)
    public_df = filtered_df[filtered_df['observerType'] == 'public']  # .sort_values(by='Date', ascending=True)
    public_html = generate_table_html(public_df, OBS_DISPLAY_COLUMNS)
    forecaster_professional_html = generate_table_html(forecaster_professional_df, OBS_DISPLAY_COLUMNS)

    # After
    filtered_df = df[(df['startDate'] >= date_object) & (df['startDate'] < week_later)]
    filtered_df = filtered_df.sort_values(by='startDate', ascending=True)
    filtered_df = filtered_df.rename(columns=OBS_DISPLAY_NAMES)
    filtered_df['Date'] = filtered_df['Date'].apply(lambda x: x.strftime('%A %Y-%m-%d'))
    forecaster_professional_df = filtered_df[filtered_df['observerType'].isin(['professional', 'forecaster'])]
    public_df = filtered_df[filtered_df['observerType'] == 'public']
    forecaster_professional_html_today = generate_table_html(forecaster_professional_df, OBS_DISPLAY_COLUMNS)
    public_html_today = generate_table_html(public_df, OBS_DISPLAY_COLUMNS)

    weather_file_path = 'weather/synoptic/BIGMS_2022_2024.csv'
    weather_df = load_data_from_s3(weather_file_path)  # Ensure this function is correctly defined to load your data
    weather_df = convert_units(weather_df)
    date_object = datetime.strptime(date, '%Y-%m-%d')

    # Filter and plot weather data
    end_date = date_object + timedelta(hours=23, minutes=59, seconds=59)  # Ensuring end of the day
    weather_df['Date_Time'] = pd.to_datetime(weather_df['Date_Time'])  # .dt.tz_localize('UTC')

    # Step 2: Convert from UTC to Mountain Standard Time (MST)
    weather_df['Date_Time'] = weather_df['Date_Time'].dt.tz_convert('America/Denver')
    weather_df['Date_Time'] = pd.to_datetime(weather_df['Date_Time']).dt.tz_localize(None)
    filtered_weather_df = weather_df[(weather_df['Date_Time'] >= start_date) & (weather_df['Date_Time'] <= end_date)]

    weather_plot_encoded = plot_weather_data(filtered_weather_df, date)

    return render_template('date_landing.html', date=formatted_date,
                           weather_plot_encoded=weather_plot_encoded,
                           forecaster_professional_observations=forecaster_professional_html,
                           public_observations=public_html,
                           forecaster_professional_html_today=forecaster_professional_html_today,
                           public_html_today=public_html_today)


@application.route('/get_yesterday_problems')
@login_required
def get_yesterday_problems():
    print('getting')
    date = request.args.get('date')
    date = date.split(' ')[1]
    date_object = datetime.strptime(date, '%Y-%m-%d')
    formatted_date = date_object.strftime('%Y-%m-%d')
    start_date = date_object - timedelta(days=7)
    day_prior = date_object - timedelta(days=1)
    day_prior_str = day_prior.strftime('%Y-%m-%d')
    # Assuming previous_forecast_rows is accessible and structured correctly
    forecast_file_path = 'forecast/fac/fac_forecast_2022_2023.csv'
    forecast_df = load_data_from_s3(forecast_file_path)
    forecast_df['problems'] = forecast_df['problems'].apply(lambda x: json.loads(x))
    previous_forecast_rows = forecast_df[forecast_df.date == day_prior_str]
    yesterday_problems_dict = {}
    for index, row in previous_forecast_rows.iterrows():
        zone = row['zone']
        problems = row['problems']  # Assuming this is already in the desired list of dicts format
        sorted_problems = sorted(problems, key=lambda x: x['problem_number'])
        yesterday_problems_dict[zone] = sorted_problems
    # Pass the HTML tables to the template

    return jsonify(yesterday_problems_dict)


@application.route('/avalanche_forecast_form/<date>')
@login_required
def avalanche_forecast_form(date):
    username = current_user.id
    filename = f'hindcast/{date}/hindcast_{username}.json'

    bucket_name = 'avalanche-data'

    # Attempt to fetch the existing form data from S3
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=filename)
        saved_data = json.loads(response['Body'].read())
    except s3_client.exceptions.NoSuchKey:
        saved_data = {}
    print(filename)
    print(saved_data)
    return render_template('avalanche_hindcast_form.html', date=date, saved_data=saved_data)


@application.route('/save_form_data', methods=['POST'])
@login_required
def save_form_data():
    data = request.json
    username = current_user.id
    date = data.get('date')  # Make sure to include 'date' in your form data
    print(f'saving data:{data}')
    if date is None:
        return jsonify({"error": "Date is required"}), 400

    file_name = f'hindcast/{date}/hindcast_{username}.json'
    bucket_name = 'avalanche-data'

    # Save the JSON data to the specified file in your S3 bucket
    s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=json.dumps(data))

    return jsonify({"message": "Data saved successfully"}), 200


@application.route('/worksheet/<date>', methods=['GET', 'POST'])
@login_required
def worksheet(date):
    username = current_user.id  # Assuming your user model has a username attribute
    file_path = f"hindcast/{date}/notes_{username}.txt"

    if request.method == 'POST':
        notes = request.form['notes']
        notes_bytes = notes.encode('utf-8')
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=file_path, Body=notes_bytes)
        flash('Saved successfully')
    else:
        # Try to load existing notes
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_path)
            notes = obj['Body'].read().decode('utf-8')
        except s3_client.exceptions.NoSuchKey:
            notes = ''
            flash("You do not have notes saved for this date.")

    return render_template('worksheet.html', date=date, name=username, notes=notes)


if __name__ == '__main__':
    application.run(debug=True)
