# Anna Jeffries, Dec. 2022

# Imports for EDA
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

# Imports for ML
from signal import signal, SIGPIPE, SIG_DFL
import pickle
import numpy as np
from flaml import AutoML
from sklearn import preprocessing

# Preliminary set up

st.set_page_config(
    page_title="CMSE 830 Project",
    # configuring layout to be wide (as opposed to centred) to to take up more screen space
    layout='wide'
)

st.header("Welcome to Anna's web app project that lets you look at information extracted from the transcripts of the iconic American TV show, *Frasier*.")

df_frasier = pd.read_csv('tidyTranscripts.csv')

tab1, tab2 = st.tabs(["Visual data exploration", "Applying the data"])

###### TAB ONE #######

with tab1:

    st.subheader(
        "Here, you can view information by season as well as by episode.")

    # columns for spacing input widgets
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(
        8)  # columns for button spacing

    with col1:
        season_select = st.selectbox(
            "Select a season:",
            (range(1, 12))
        )

    with col5:
        episode_select = st.selectbox(
            "Which episode of season " + str(season_select) + "?",
            (range(1, 25))
        )

    # word count for every episode orgd by season
    df_frasier_totalwords = df_frasier.groupby([
        'season', 'episode', 'characterName', 'gender', 'actorName', 'characterType', 'episodeCount',
        'originalAirDate', 'title', 'viewershipInMillions', 'imdbRatings']).size().reset_index(name='total_words')

    # # word count organized by character for every episode orgd by season
    # df_frasier_characterwords = df_frasier.groupby([
    #     'season', 'episode', 'characterName', 'gender', 'actorName', 'characterType', 'episodeCount',
    #     'originalAirDate', 'title', 'viewershipInMillions', 'imdbRatings']).size().reset_index(name='total_words')

    # functions to get the season and episode info
    def get_season(season):
        '''selecting the entries matching the season selected'''
        return df_frasier_totalwords.loc(df_frasier_totalwords['season'] == season)

    def get_season_words():
        '''this is simply summing the words in a given season and returning as list'''
        sums = []
        for i in range(1, 12):
            ss = sum(df_frasier_totalwords.loc(
                df_frasier_totalwords['season'] == i)['total_words'])
            sums.append(ss)
        return sums

    def get_episode(episode, the_season):
        '''takes 2 parameters because the season selected needs to be used to select the right episode'''
        return the_season.where(the_season['episode'] == episode).dropna()

    # calling functions to create the df needed for plotting
    df_season = get_season(season_select)
    df_episode = get_episode(episode_select, df_season)

    # getting the list of season word totals as well as a list of just the seasons and adding them to the main DataFrame called in season_plot because Altair is freaking annoying and refuses to take in anything other than the one dataset so I'm having to add awkward hanging columns to get the information into the chart that I need to have....
    season_words = get_season_words()
    season_totals = pd.DataFrame({'totals': season_words})
    df_frasier_totalwords['season_total'] = season_totals
    df_frasier_totalwords['seasonlist'] = pd.DataFrame(range(1, 12))

    rating_avg = []
    for i in range(1, 12):
        rating_avg.append(mean(df_frasier_totalwords['imdbRatings'].loc(
            df_frasier_totalwords['season'] == i)))

    viewing_avg = []
    for i in range(1, 12):
        viewing_avg.append(mean(df_frasier_totalwords['viewershipInMillions'].loc(
            df_frasier_totalwords['season'] == i)))

    df_frasier_totalwords['rating_avg'] = pd.Series(
        [round(num, 2) for num in rating_avg])
    df_frasier_totalwords['viewing_avg'] = pd.Series(
        [round(num, 2) for num in viewing_avg])

    df_frasier_totalwords['rating_avg'] = pd.Series(
        [round(num, 2) for num in rating_avg])
    df_frasier_totalwords['viewing_avg'] = pd.Series(
        [round(num, 2) for num in viewing_avg])

    # plotting the season chart with total words by season
    season_plot = alt.Chart(df_frasier_totalwords, padding={'left': 0, 'top': 25, 'right': 0, 'bottom': 5}).mark_bar(size=30).encode(
        x=alt.X('seasonlist', axis=alt.Axis(title='Season', grid=False)),
        y=alt.Y('season_total', axis=alt.Axis(title='Total number of words')),
        color=alt.condition(
            alt.datum.seasonlist == season_select,
            alt.value('crimson'),
            alt.value('darkgrey')),
        tooltip=['season_total']
    ).configure_view(strokeWidth=0).properties(width=600).interactive()

    # plotting the episode chart with words by character
    episode_plot = alt.Chart(df_episode, padding={'left': 0, 'top': 25, 'right': 5, 'bottom': 5}
                             ).mark_bar(size=35).encode(
        x=alt.X('characterName', axis=alt.Axis(title='Characters')),
        y=alt.Y('total_words', axis=alt.Axis(title='Total number of words')),
        color=alt.Color('characterName',
                        scale=alt.Scale(scheme='turbo'), legend=alt.Legend(title='Characters', orient='right')),
        tooltip=['total_words', 'actorName', 'characterType', 'gender']
    ).configure_view(strokeWidth=0).properties(width=alt.Step(50)).interactive()

    # columns for spacing plots
    col1a, col2a = st.columns(2)

    with col1a:
        season_plot
    with col2a:
        episode_plot

    # SECOND SET OF PLOTS

    st.subheader(
        "Here, you can view information by character in a given season.")

    # List of all main characters by character name
    main_ch = df_frasier_totalwords.loc[df_frasier_totalwords['characterType'] == 'main']
    main_ch_names = list(main_ch['characterName'].unique())
    # List of all recurring characters by character name
    recur_ch = df_frasier_totalwords.loc[df_frasier_totalwords['characterType'] == 'recurring']
    recur_ch_names = list(recur_ch['characterName'].unique())

    with st.container():

        col1b, col2b, col3b, col4b, col5b, col6b, col7b, col8b = st.columns(8)

        with col1b:
            ch_select = st.selectbox(
                "Select a main character:",
                main_ch_names
            )

        with col2b:
            ch_season_select = st.selectbox(
                "Which season?",
                (range(1, 12))
            )

        def get_ch_season(ch_season, char):
            '''selecting the entries matching the character name and season selected'''
            df = df_frasier_totalwords.loc[df_frasier_totalwords['season'] == ch_season]
            return df.loc[df['characterName'] == char]

        # calling functions to create the df needed for plotting
        df_ch_season = get_ch_season(ch_season_select, ch_select)

        col1c, col2c, col3c = st.columns(3)

        ch_season_plot = alt.Chart(df_ch_season, padding={'left': 0, 'top': 25, 'right': 0, 'bottom': 5}).mark_line(
            color='gold', point=alt.OverlayMarkDef(size=80), width=15).encode(
            x=alt.X('episode', axis=alt.Axis(title='Episodes', grid=False)),
            y=alt.Y('total_words', axis=alt.Axis(
                title='Total number of words')),
            tooltip=['title', 'total_words', 'actorName', 'gender']
        ).configure_view(strokeWidth=0).properties(width=450).interactive()

        with col1c:
            ch_season_plot

        with col2c:
            achoice = st.checkbox(
                'I would like to compare ' + ch_select + ' with another main character')

            if achoice:

                ch1_select = st.selectbox(
                    "Select a main character:", key='secondarymain', options=main_ch_names)

                df_ch1_season = get_ch_season(ch_season_select, ch1_select)
                df_ch1_season_combo = pd.concat(
                    [df_ch_season, df_ch1_season], ignore_index=True)

                ch_season_combo_plot = alt.Chart(df_ch1_season_combo, padding={'left': 0, 'top': 25, 'right': 0, 'bottom': 5}).mark_line(
                    point=alt.OverlayMarkDef(size=50), width=10).encode(
                    x=alt.X('episode', axis=alt.Axis(
                        title='Episodes', grid=False)),
                    y=alt.Y('total_words', axis=alt.Axis(
                        title='Total number of words')),
                    color=alt.Color('characterName', scale=alt.Scale(scheme='rainbow'),
                                    legend=alt.Legend(title='Characters', orient='bottom')),
                        tooltip=['title', 'total_words', 'actorName', 'gender']
                ).configure_view(strokeWidth=0).properties(width=450).interactive()

                ch_season_combo_plot

        with col3c:
            anotherchoice = st.checkbox(
                'I would like to compare ' + ch_select + ' with a recurring character', key='recurring')

            if anotherchoice:

                ch_recur_select = st.selectbox(
                    "Select a recurring character:", key='secondaryrecur', options=recur_ch_names)

                df_ch_rec_season = get_ch_season(
                    ch_season_select, ch_recur_select)
                df_chrec_season_combo = pd.concat(
                    [df_ch_season, df_ch_rec_season], ignore_index=True)

                df_chrec_season_combo_plot = alt.Chart(df_chrec_season_combo, padding={'left': 0, 'top': 25, 'right': 0, 'bottom': 5}).mark_line(
                    point=alt.OverlayMarkDef(size=50), width=10).encode(
                    x=alt.X('episode', axis=alt.Axis(
                        title='Episodes', grid=False)),
                    y=alt.Y('total_words', axis=alt.Axis(
                        title='Total number of words')),
                    color=alt.Color('characterName', scale=alt.Scale(scheme='rainbow'),
                                    legend=alt.Legend(title='Characters', orient='bottom')),
                    tooltip=['title', 'total_words', 'actorName', 'gender']
                ).configure_view(strokeWidth=0).properties(width=450).interactive()
                df_chrec_season_combo_plot

    # THIRD SET

    st.subheader(
        "And finally, you can view information by character across the entire show.")

    with st.container():

        col1d, col2d, col3d, col4d = st.columns(4)

        with col1d:
            ch_options = st.multiselect('Select as many characters as you please',
                                        main_ch_names + recur_ch_names, ['Frasier Crane'])

        def get_ch_show(char):
            '''selecting the entries matching the character name'''
            return df_frasier_totalwords.loc[df_frasier_totalwords['characterName'] == char]

        df_selection_show = pd.DataFrame()
        for person in ch_options:
            df_selection_show = pd.concat(
                [df_selection_show, get_ch_show(person)])

        ch_show_plot = alt.Chart(df_selection_show).mark_line(point=alt.OverlayMarkDef(size=30), width=5).encode(
            x=alt.X('episodeCount', axis=alt.Axis(
                title='Episodes by cumulative count', grid=False)),
            y=alt.Y('total_words', axis=alt.Axis(
                title='Total number of words')),
            color=alt.Color('characterName', scale=alt.Scale(scheme='rainbow'), legend=alt.Legend(
                title='Characters')),
            tooltip=['total_words', 'title'],
        ).configure_view(strokeWidth=0).properties(height=500, width=1400)

        # areas = alt.Chart(seasons_rect.reset_index()).mark_rect(opacity=0.3).encode(
        #         x='start', x2='stop',
        #         color=alt.Color('index:N',scale=alt.Scale(scheme='sinebow'),
        #         legend=alt.Legend(title='Seasons'))).properties(height=500,width=1400)

        ch_show_plot

########### TAB TWO ############

### TRAINING ML ###

# sourced from Stack Overflow; to stop exceptions so ML can complete training without raising errors
signal(SIGPIPE, SIG_DFL)

# Characters df
characters_df = df_frasier
characters_df.loc[characters_df["characterType"]
                  == "other", "characterName"] = "Other"
characters_df = (
    characters_df[["characterName", "episodeCount"]]
    .groupby(
        [
            "characterName",
            "episodeCount",
        ]
    )
    .size()
    .unstack("characterName")
    .fillna(0)
)

# Gender df
gender_df = df_frasier
gender_df["female"] = gender_df["gender"] == "female"
gender_df = (
    gender_df[["episodeCount", "female"]]
    .groupby("episodeCount")
    .mean()
)

# Viewership (millions) df
viewership_df = (
    df_frasier[["episodeCount", "viewershipInMillions"]]
    .groupby("episodeCount")
    .first(numeric_only=True)
)

# Writers df
writers_df = (
    df_frasier[["episodeCount", "writtenBy"]]
    .groupby("episodeCount")
    .first()
)

# Directors df
directors_df = (
    df_frasier[["episodeCount", "directedBy"]]
    .groupby("episodeCount")
    .first()
)

# Putting together all the dfs
feat_df = pd.concat([gender_df, viewership_df, characters_df,
                    writers_df, directors_df], axis=1)


# IMDB Ratings df (Y)
ratings = (
    df_frasier[["imdbRatings", "episodeCount"]]
    .groupby("episodeCount")
    .first()
)

# ML dfs

# splitting ratings (Y): setting aside season 11 episodes for testing
season_11_y_test = ratings.tail(24)
# the training set
rates_y_train = ratings[:-24]

# splitting features (X): setting aside season 11 episodes
season_11_x_test = feat_df.tail(24)
# the training set
feat_df_x_train = feat_df[:-24]

# Scaling X train

# Getting episodeCount index and saving it to add back later
ep_index_train = feat_df_x_train.reset_index()
ep_count_train = ep_index_train['episodeCount']

# Making a copy of the training df with numeric columns only to scale; saving column names
numeric_x_train = feat_df_x_train.drop(['directedBy', 'writtenBy'], axis=1)
col_names_train = numeric_x_train.columns

# Scaling
a_scaler = preprocessing.StandardScaler()
a_scaler.fit(numeric_x_train)
X_train_scaled_temp = a_scaler.transform(numeric_x_train)
# Adding categorical columns back in with episodeCount
X_train_scaled = pd.DataFrame(X_train_scaled_temp, index=ep_count_train)
# adding column names back in
X_train_scaled.columns = col_names_train
X_train_scaled['directedBy'] = feat_df_x_train['directedBy']
X_train_scaled['writtenBy'] = feat_df_x_train['writtenBy']

# Repeating process for scaling X test

# Getting episodeCount index and saving it to add back later
ep_index_test = season_11_x_test.reset_index()
ep_count_test = ep_index_test['episodeCount']

# numeric X test
numeric_x_test = season_11_x_test.drop(['directedBy', 'writtenBy'], axis=1)
col_names_test = numeric_x_test.columns

# Scaling X test after fitting on X train
X_test_scaled_temp = a_scaler.transform(numeric_x_test)
# adding categorical columns back in and setting index to episodeCount
X_test_scaled = pd.DataFrame(X_test_scaled_temp, index=ep_count_test)
# adding column names back in
X_test_scaled.columns = col_names_test
X_test_scaled['directedBy'] = season_11_x_test['directedBy']
X_test_scaled['writtenBy'] = season_11_x_test['writtenBy']

# the model!
automl = AutoML()
automl_settings = {
    "time_budget": 1,
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": "frasier.log",
}

y_train = np.array(rates_y_train)

# Train with labeled input data
automl.fit(X_train=X_train_scaled, y_train=y_train,
           **automl_settings)

# Save the model
with open("automl.pkl", "wb") as f:
    pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)


with tab2:

    with open("automl.pkl", "rb") as f:
        automl = pickle.load(f)
    pred = automl.predict(X_test_scaled)

    rate_compare = season_11_y_test.reindex(
        list(range(1, 49))).set_index(np.tile(ep_count_test, 2))
    rate_compare['imdbRatings'] = np.append(
        season_11_y_test['imdbRatings'], pred)
    rate_compare['type'] = np.append(
        np.tile('Actual', 24), np.tile('Predicted', 24))

    # X_pred_info = pd.DataFrame()
    # X_pred_info['Features'] = automl.feature_names_in_
    # X_pred_info['Importances'] = automl.feature_importances_

    # auto_chart = alt.Chart(X_pred_info).mark_bar.encode(
    #     x=alt.X('Features', axis=alt.Axis(
    #         title='Features', grid=False)),
    #     y=alt.Y('Importances', axis=alt.Axis(title='Impacts')),
    #     color=alt.Color('characterName', scale=alt.Scale(scheme='rainbow'), legend=alt.Legend(
    #         title='Characters')),
    # ).configure_view(strokeWidth=0).properties(height=500, width=1400)

    actual_chart = alt.Chart(rate_compare).mark_bar.encode(
        x=alt.X('episodeCount', axis=alt.Axis(
            title='Episode count', grid=False)),
        y=alt.Y('predicted', axis=alt.Axis(title='Rating')),
        column='type:N',
        color=alt.Color('type', scale=alt.Scale(scheme='rainbow')),
    ).configure_view(strokeWidth=0).properties(height=500, width=1400)

    st.subheader(
        "Results of the model's IMDB rating predictions vs. the actual IMDB ratings.")

    actual_chart
