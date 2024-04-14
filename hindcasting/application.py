# Standard Library Imports
from datetime import datetime, timedelta
import json
import io
import base64
import random

# Third-party Package Imports
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash
import boto3
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates

from constants import S3_BUCKET_NAME, OBS_DISPLAY_NAMES, OBS_DISPLAY_COLUMNS, \
    AV_DISPLAY_COLUMNS, aai_weights, station_colors
from s3_helpers import load_data_from_s3, get_completed_dates, write_completed_dates, get_s3_object, put_s3_object
from users import users

application = Flask(__name__)
application.secret_key = 'k'

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(application)

# Constants
pd.options.mode.chained_assignment = None


def compute_aai(rows):
    if len(rows) == 0:
        aai = 0
    else:
        rows['dSize'] = rows['dSize'].replace('Unk Size', 1).astype(float).apply(lambda x: int(x))

        # Compute weighted sum
        rows['weighted'] = rows['dSize'].map(aai_weights)
        rows['weighted'] = rows['weighted'] * rows['number']
        aai = rows['weighted'].sum()
    return aai


def convert_units(df):
    # Celsius to Fahrenheit
    df['air_temp_set_1'] = df['air_temp_set_1'] * 9 / 5 + 32
    # m/s to mph
    df['wind_speed_set_1'] = df['wind_speed_set_1'] * 2.23694
    # mm to inches
    df['precip_accum_one_hour_set_1'] = df['precip_accum_one_hour_set_1'] / 25.4
    return df


def plot_weather_data(all_data, date):
    """Generate plots for temperature, wind speed, and accumulated precipitation, for each station."""
    date = pd.to_datetime(date)
    if pd.to_datetime(date) < pd.to_datetime('2023-10-01'):
        season_start_date = pd.to_datetime('2022-10-15')
    else:
        season_start_date = pd.to_datetime('2023-10-15')
    all_data['Date_Time'] = pd.to_datetime(all_data['Date_Time'])
    start_date = date - timedelta(days=14)
    end_date = date + timedelta(days=1, hours=23, minutes=59, seconds=59)
    df = all_data[(all_data['Date_Time'] >= start_date) & (all_data['Date_Time'] <= end_date)]
    season_df = all_data[(all_data['Date_Time'] >= season_start_date) & (all_data['Date_Time'] <= end_date)]
    idxs = {'STAM8':0,'NOISY':1,'FTMM8':2}
    df['date'] = df['Date_Time'].dt.date
    df['wind_direction'] = df['wind_direction'].astype(float)

    # column for AM/PM to represent 12-hour intervals
    df['interval'] = df['Date_Time'].dt.hour // 12
    df['interval_id'] = df['date'].astype(str) + "_" + df['interval'].astype(str)

    df['wind_dir_rad'] = np.deg2rad(df['wind_direction'])
    df['u_component'] = np.cos(df['wind_dir_rad'])
    df['v_component'] = np.sin(df['wind_dir_rad'])

    interval_avg = df.groupby(['Station_ID', 'interval_id']).agg({
        'u_component': 'mean',
        'v_component': 'mean',
        'wind_speed_mph': 'mean'
    }).reset_index()

    interval_avg['avg_wind_dir_rad'] = np.arctan2(interval_avg['v_component'], interval_avg['u_component'])
    interval_avg['avg_wind_dir'] = np.rad2deg(interval_avg['avg_wind_dir_rad']) % 360

    daily_df = df.groupby(['Station_ID', 'date']).agg({
        'precip_accum_in': 'last',
    }).reset_index()
    daily_df['daily_diff'] = daily_df.groupby('Station_ID')['precip_accum_in'].diff().fillna(0)
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))
    stations = df['Station_ID'].dropna().unique()

    legend_added = set()  # To keep track of legends already added
    bar_width = 0.22
    for idx, station in enumerate(stations):
        station_df = df[df['Station_ID'] == station].sort_values('Date_Time')
        station_daily_df = daily_df[daily_df['Station_ID'] == station]
        station_season_df = season_df[season_df.Station_ID == station].sort_values('Date_Time')
        interval_avg_station = interval_avg[interval_avg['Station_ID'] == station]

        color = station_colors.get(station, 'black')

        axs[0].plot(station_df['Date_Time'], station_df['air_temp_F'], label=station, color=color)
        if station not in ['FTMM8']:
            axs[1].plot(station_df['Date_Time'], station_df['wind_speed_mph'], label=station, color=color)

            for _, row in interval_avg_station.iterrows():
                interval_start = pd.to_datetime(row['interval_id'].split('_')[0])
                # Determine AM/PM from interval_id
                if row['interval_id'].endswith("_1"):  # PM
                    interval_start += pd.Timedelta(hours=12)
                u = np.sin(row['avg_wind_dir_rad'])
                v = np.cos(row['avg_wind_dir_rad'])
                if station not in legend_added:  # Add legend only once per station
                    axs[2].quiver(interval_start, 0, u, v,
                                  color=color, label=station, headlength=0.1, headwidth=0.1, headaxislength=0.1,
                                  width=0.002)
                    legend_added.add(station)
                else:
                    axs[2].quiver(interval_start, 0, u, v,
                                  color=color, headlength=0.1, headwidth=0.1, headaxislength=0.1, width=0.002)

        if np.sum(station_df['precip_accum_in']) > 0:
            if station != 'HORNET':
                axs[3].plot(station_df['Date_Time'],
                            station_df['precip_accum_in'].values - station_df['precip_accum_in'].iloc[0],
                            label=station, color=color)
                if station in ['STAM8','NOISY','FTMM8']:
                # Plotting daily precipitation difference as bars
                    new_dates = [
                        mdates.date2num(date) + .25 + .25*idxs[station] for date in station_daily_df['date']]
                    print(new_dates)
                    axs[3].bar(new_dates, station_daily_df['daily_diff'], color=color, width=bar_width, label=station+' Daily Delta')

        if np.sum(station_df['snow_water_equiv_in']) > 0:
            axs[4].plot(station_season_df['Date_Time'], station_season_df['snow_water_equiv_in'], label=station,
                        color=station_colors[station])
    axs[0].axhline(y=32, color='blue', linestyle='--', label='Freezing point')  # Add this line
    # Titles
    axs[0].set_title('Station Temperature Last 2 Weeks')
    axs[1].set_title('Station Wind Speed Last 2 Weeks (mph)')
    axs[2].set_title('Vector Average Station Wind Direction Last 2 Weeks (12 Hour Intervals)')
    axs[3].set_title('Net Accumulated Precipitation Last 2 Weeks (in)')
    axs[4].set_title('Snow Water Equivalent Season to Date (in)')

    axs[0].set_ylabel('Temperature (F)')
    axs[3].set_ylabel('Precip Accum (in)')
    axs[4].set_ylabel('SWE (in)')
    axs[2].yaxis.set_visible(False)
    axs[1].set_ylabel('Wind Speed (mph)')

    for ax in axs[0:-1]:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    for ax in axs:
        ax.axvline(pd.to_datetime(date), color='r', linestyle='--', label='Forecast Date Start and Stop')
        ax.axvline(pd.to_datetime(date) + timedelta(days=1), color='r', linestyle='--', )
        ax.legend(ncol=3)
        ax.tick_params(axis='x', rotation=35)
        ax.grid('on')

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf8')


def plot_avalanche_data(df, date, obs_df):
    date = pd.to_datetime(date)
    start_x_date = date - timedelta(days=7)
    end_x_date = date + timedelta(days=10)

    df['date'] = pd.to_datetime(df['date']).apply(lambda x: x.replace(tzinfo=None))
    df.loc[df['dSize'] == 'Unk Size', 'dSize'] = 1
    colors = {1.0: 'blue', 2.0: 'green', 3.0: 'yellow', 4.0: 'red'}

    # Separate into known and unknown date
    kdf = df[df.dateKnown == True]
    idf = df[df.dateKnown == False & df.delta_obs_days.notnull()]
    idf = idf[idf.delta_obs_days.notnull()]

    # Process inexact dates
    idf['early_date'] = idf['date']
    idf['late_date'] = idf.apply(lambda x: x['date'] + timedelta(days=min(x['delta_obs_days'], 3000)), axis=1)
    idf['dSize'] = idf['dSize'].apply(lambda x: np.floor(float(x)))

    grp = idf.groupby(['date', 'late_date', 'early_date', 'dSize'], as_index=False).agg({'number': 'sum'})
    grp.sort_values(by=['date', 'dSize'], inplace=True)
    kdf.sort_values(by=['date', 'dSize'], inplace=True)

    obs_df = obs_df[(obs_df['startDate'] >= start_x_date) & (obs_df['startDate'] <= end_x_date)]
    obs_df['startDate'] = obs_df['startDate'].apply(lambda x: x.strftime('%Y-%m-%d'))
    obs_df['is_public'] = obs_df['is_professional'].apply(lambda x: True if x == False else False)
    obs_grouped = obs_df.groupby(['startDate', 'is_public']).size().unstack(fill_value=0)
    y_offset = {date: random.uniform(0, .05) for date in grp['date'].unique()}
    y_offset_kdf = {date: 0 for date in kdf['date'].unique()}

    fig, ax = plt.subplots(3, 1, figsize=(10, 13))

    for _, row in grp.iterrows():
        left = mdates.date2num(row['date'])
        height = row['number']
        color = colors[int(float(row['dSize']))]
        bottom = y_offset[row['date']]

        ax[1].bar(left, height, 0.8, bottom=bottom, color=color, align='center')
        y_offset[row['date']] += height

        # Error bars
        early_date_num = mdates.date2num(row['early_date'])
        late_date_num = mdates.date2num(row['late_date'])
        # ax[1].errorbar(x=left, y=bottom + height / 2, xerr=[[left - early_date_num], [late_date_num - left]],
        #                fmt='none', ecolor='black', capsize=5)
        _, caplines, _ = ax[1].errorbar(x=left, y=bottom + height / 2,
                                        xerr=[[left - early_date_num], [late_date_num - left]],
                                        fmt='none', ecolor='black', capsize=0)  # capsize set to 0

        capsize_left = 0.01  # Small cap size for left
        capsize_right = .75  # Large cap size for right

        # Coordinates for left cap
        left_cap_x = [left - (left - early_date_num), left - (left - early_date_num)]
        left_cap_y = [bottom + height / 2 - capsize_left / 2, bottom + height / 2 + capsize_left / 2]

        # Coordinates for right cap
        right_cap_x = [left + (late_date_num - left), left + (late_date_num - left)]
        right_cap_y = [bottom + height / 2 - capsize_right / 2, bottom + height / 2 + capsize_right / 2]

        ax[1].plot(left_cap_x, left_cap_y, color='black')
        ax[1].plot(right_cap_x, right_cap_y, color='black')

    # Plotting for known dates
    for _, row in kdf.iterrows():
        left = mdates.date2num(row['date'])
        height = row['number']
        color = colors[int(float(row['dSize']))]
        bottom = y_offset_kdf[row['date']]

        ax[0].bar(left, height, 0.8, bottom=bottom, color=color, align='center')
        y_offset_kdf[row['date']] += height

    obs_grouped.plot(kind='bar', stacked=True, ax=ax[2], color=['skyblue', 'orange'])
    ax[2].set_title('Observations by Date')
    ax[2].set_ylabel('Number of Observations')
    ax[2].legend(title='Observation Type', labels=['Professional', 'Public'])

    for i, title in enumerate(['Exact Date Avalanche Activity', 'Inexact Date Avalanche Activity']):
        legend_patches = [Patch(color=color, label=f'dSize: {size}') for size, color in colors.items()]
        ax[i].legend(handles=legend_patches, title="Legend")
        ax[i].set_title(title)

    for a in ax[0:2]:
        a.set_xlim([start_x_date, end_x_date])
        a.xaxis_date()
        a.xaxis.set_major_locator(mdates.DayLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    for a in ax:
        a.tick_params(axis='x', rotation=45)

        for label in a.get_xticklabels():
            if pd.to_datetime(label.get_text()) == date:
                label.set_color('red')
                label.set_weight('bold')
    max_height_kdf = max([y_offset_kdf[date] for date in y_offset_kdf if
                          start_x_date <= date <= end_x_date], default=0)
    ax[0].set_ylim(0, max_height_kdf * 1.1)
    max_height_idf = max(
        [y_offset[date] for date in y_offset if start_x_date <= date <= end_x_date],
        default=0)
    ax[1].set_ylim(0, max_height_idf * 1.1)  # Adjust the 1.1 as needed to provide some padding above the tallest bar

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf8')


def generate_dropdown(row_id):
    return f'''<select name="usefulness_{row_id}" class="usefulness-dropdown">
        <option value="not_used">Not Used</option>
        <option value="slightly_useful">Slightly Useful</option>
        <option value="moderately_useful">Moderately Useful</option>
        <option value="very_useful">Very Useful</option>
    </select>'''


def assign_row_class(row, date):
    if row['startDate'] == date or date in row['avalanche_link_text']:
        return 'highlight-row'
    else:
        return 'normal-row'


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
        row_class = row['row_class']
        html += f'<tr class="{row_class}">'

        for col in columns:
            cell_value = row[col]
            if isinstance(cell_value, str) and (cell_value.startswith('<a ') or cell_value.startswith('<div ')):
                cell_html = cell_value
            else:
                cell_html = f'{cell_value}'
            html += f'<td>{cell_html}</td>'

        html += '</tr>'

    html += '</tbody></table>'
    return html


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
    completed_dates = get_completed_dates(current_user.id)
    dates_with_status = [
        {'date': date, 'completed': date in completed_dates}
        for date in user_dates
    ]

    return render_template('user_home.html', dates=dates_with_status, user=current_user.name)


@application.route('/date_landing/<date>')
@login_required
def date_landing(date):
    date_object = datetime.strptime(date, '%Y-%m-%d')
    if pd.to_datetime(date_object) >= pd.to_datetime('2023-10-15'):
        season = 2024
    else:
        season = 2023
    formatted_date = date_object.strftime('%A %Y-%m-%d')
    username = current_user.id
    filename = f'hindcast/{formatted_date}/mainpage_{username}.json'
    saved_data = get_s3_object(filename)

    if date not in users[current_user.id]['dates']:
        flash("You do not have access to this date.")
        return redirect(url_for('user_home'))

    formatted_date_only = date_object.strftime('%Y-%m-%d')
    start_date = date_object - timedelta(days=7)
    week_later = date_object + timedelta(days=7)
    # Load data from S3
    file_path = f'observations/general/fac_{int(season - 1)}_{season}.csv'
    df = load_data_from_s3(file_path)
    df['row_class'] = df.apply(assign_row_class, axis=1, args=(formatted_date_only,))
    df['Useful?'] = df['id'].apply(generate_dropdown)
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['snowpack_description'] = df['snowpack_description'].apply(
        lambda x: f'<div class="scrollable-content">{x}</div>')
    df['link'] = df['link'].apply(
        lambda x: f'<a href="{x}" target="_blank">View</a>')
    df['is_professional'] = df['observerType'].isin(['professional', 'forecaster'])
    filtered_df = df[(df['startDate'] > start_date) & (df['startDate'] < date_object)]
    filtered_df = filtered_df.sort_values(by='startDate', ascending=False)
    filtered_df = filtered_df.rename(columns=OBS_DISPLAY_NAMES)
    filtered_df['Date'] = filtered_df['Date'].apply(lambda x: x.strftime('%A %Y-%m-%d'))
    #  Forecaster/Professional and Public observations
    forecaster_professional_df = filtered_df[
        filtered_df.is_professional == True]
    public_df = filtered_df[filtered_df.is_professional == False]  # .sort_values(by='Date', ascending=True)
    public_html = generate_table_html(public_df, OBS_DISPLAY_COLUMNS)
    forecaster_professional_html = generate_table_html(forecaster_professional_df, OBS_DISPLAY_COLUMNS)

    # After
    filtered_df = df[(df['startDate'] >= date_object) & (df['startDate'] < week_later)]
    filtered_df = filtered_df.sort_values(by='startDate', ascending=True)
    filtered_df = filtered_df.rename(columns=OBS_DISPLAY_NAMES)
    filtered_df['Date'] = filtered_df['Date'].apply(lambda x: x.strftime('%A %Y-%m-%d'))
    forecaster_professional_df = filtered_df[filtered_df.is_professional == True]
    public_df = filtered_df[
        filtered_df.is_professional == False]
    forecaster_professional_html_today = generate_table_html(forecaster_professional_df, OBS_DISPLAY_COLUMNS)
    public_html_today = generate_table_html(public_df, OBS_DISPLAY_COLUMNS)

    weather_file_path = 'weather/synoptic/all_v2_2022_2024.csv'
    weather_df = load_data_from_s3(weather_file_path)

    # Filter and plot weather data
    weather_df['Date_Time'] = pd.to_datetime(weather_df['Date_Time'])  # .dt.tz_localize('UTC')
    weather_plot_encoded = plot_weather_data(weather_df, date)

    avalanche_file_path = f'observations/avalanche/fac_{int(season - 1)}_{season}.csv'
    avalanche_df = load_data_from_s3(avalanche_file_path)
    filtered_df = avalanche_df[avalanche_df.date == formatted_date_only]
    filtered_df['row_class'] = 'normal-row'
    avalanche_html_today = generate_table_html(filtered_df, AV_DISPLAY_COLUMNS)
    aai = compute_aai(filtered_df)
    avalanche_plot_encoded = plot_avalanche_data(avalanche_df, date, df)
    return render_template('date_landing.html', date=formatted_date,
                           weather_plot_encoded=weather_plot_encoded,
                           forecaster_professional_observations=forecaster_professional_html,
                           public_observations=public_html,
                           forecaster_professional_html_today=forecaster_professional_html_today,
                           public_html_today=public_html_today,
                           avalanche_html_today=avalanche_html_today,
                           aai=aai,
                           saved_data=saved_data,
                           avalanche_plot_encoded=avalanche_plot_encoded)


@application.route('/get_yesterday_problems')
@login_required
def get_yesterday_problems():
    date = request.args.get('date')
    date = date.split(' ')[1]
    date_object = datetime.strptime(date, '%Y-%m-%d')
    if pd.to_datetime(date_object) >= pd.to_datetime('2023-10-15'):
        season = 2024
    else:
        season = 2023
    day_prior = date_object - timedelta(days=1)
    day_prior_str = day_prior.strftime('%Y-%m-%d')
    forecast_file_path = f'forecast/fac/fac_forecast_{int(season - 1)}_{season}.csv'
    forecast_df = load_data_from_s3(forecast_file_path)
    forecast_df['problems'] = forecast_df['problems'].apply(lambda x: json.loads(x))
    previous_forecast_rows = forecast_df[forecast_df.date == day_prior_str]
    yesterday_problems_dict = {}
    for index, row in previous_forecast_rows.iterrows():
        zone = row['zone']
        problems = row['problems']
        sorted_problems = sorted(problems, key=lambda x: x['problem_number'])
        yesterday_problems_dict[zone] = sorted_problems

    return jsonify(yesterday_problems_dict)


@application.route('/avalanche_forecast_form/<date>')
@login_required
def avalanche_forecast_form(date):
    username = current_user.id
    filename = f'hindcast/{date}/hindcast_{username}.json'

    saved_data = get_s3_object(filename)
    return render_template('avalanche_hindcast_form.html', date=date, saved_data=saved_data)


@application.route('/save_dropdown_data', methods=['POST'])
@login_required
def save_dropdown_data():
    data = request.json
    username = current_user.id
    date = data.get('date')
    if date is None:
        return jsonify({"error": "Date is required"}), 400
    filename = f'hindcast/{date}/mainpage_{username}.json'
    put_s3_object(json.dumps(data), filename)
    return jsonify({"message": "Data saved successfully"}), 200


@application.route('/save_form_data', methods=['POST'])
@login_required
def save_form_data():
    data = request.json
    username = current_user.id
    date = data.get('date')
    print(f'saving data:{data}')
    if date is None:
        return jsonify({"error": "Date is required"}), 400

    file_name = f'hindcast/{date}/hindcast_{username}.json'
    put_s3_object(json.dumps(data), file_name)

    return jsonify({"message": "Data saved successfully"}), 200


@application.route('/worksheet/<date>', methods=['GET', 'POST'])
@login_required
def worksheet(date):
    username = current_user.id
    filename = f"hindcast/{date}/notes_{username}.txt"

    if request.method == 'POST':
        notes = request.form['notes']
        notes_bytes = notes.encode('utf-8')
        put_s3_object(notes_bytes, filename)
        flash('Saved successfully')
    else:
        notes = get_s3_object(filename, format='utf-8', default='')

    return render_template('worksheet.html', date=date, name=username, notes=notes)


@application.route('/submit_date', methods=['POST'])
def submit_date():
    data = request.json
    user_id = current_user.id
    date = data.get('date')
    try:
        parsed_date = datetime.strptime(date, '%A %Y-%m-%d')
    except ValueError as e:
        return jsonify({"error": f"Invalid date format: {str(e)}"}), 400

    new_date = parsed_date.strftime('%Y-%m-%d')

    try:
        completed_dates = get_completed_dates(user_id)

        if new_date not in completed_dates:
            completed_dates.append(new_date)
            write_completed_dates(user_id, completed_dates)

        return jsonify({"message": "Date submitted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    application.run(debug=True)
