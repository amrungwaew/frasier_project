# Anna Jeffries, Dec. 2022

# Imports for EDA
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st
from PIL import Image

# Imports for ML
# from signal import signal, SIGPIPE, SIG_DFL
import pickle
import numpy as np
from flaml import AutoML
from flaml.ml import sklearn_metric_loss_score
from sklearn import preprocessing

# Preliminary set up

st.set_page_config(
    page_title="CMSE 830 Project",
    # configuring layout to be wide (as opposed to centred) to to take up more screen space
    layout='wide'
)

header = Image.open('heading.jpeg')
st.image(header)

st.header("Welcome to Anna's web app project that lets you look at information extracted from the transcripts of the iconic American TV show, *Frasier*.")
df_frasier = pd.read_csv('tidyTranscripts.csv')

tab1, tab2, tab3, tab4 = st.tabs(
    ["Visual data exploration", "Applying the data", "Miscellaneous", "Notes for Murillo"])

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
        return df_frasier_totalwords.where(df_frasier_totalwords['season'] == season).dropna()

    def get_season_words():
        '''this is simply summing the words in a given season and returning as list'''
        sums = []
        for i in range(1, 12):
            ss = sum(df_frasier_totalwords.where(
                df_frasier_totalwords['season'] == i).dropna()['total_words'])
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
        rating_avg.append(mean(df_frasier_totalwords['imdbRatings'].where(
            df_frasier_totalwords['season'] == i).dropna()))

    viewing_avg = []
    for i in range(1, 12):
        viewing_avg.append(mean(df_frasier_totalwords['viewershipInMillions'].where(
            df_frasier_totalwords['season'] == i).dropna()))

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
# signal(SIGPIPE, SIG_DFL)

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
#### ML dfs ####

# splitting ratings (Y): setting aside season 11 episodes for testing
season_11_y_test = ratings.tail(24)
# the training set
rates_y_train = ratings[:-24]


def split_y_test(season):
    temp = [x+((season-1)*24) for x in range(1, 25)]
    season_test = pd.DataFrame()
    return pd.concat([season_test, ratings[temp[0]-1:temp[-1]]])


def split_y_train(season):
    '''splitting df to get y train'''
    temp = [x+((season-1)*24) for x in range(1, 25)]
    return ratings.drop(ratings.index[temp[0]-1:temp[-1]])


# splitting features (X): setting aside season 11 episodes
season_11_x_test = feat_df.tail(24)
# the training set
feat_df_x_train = feat_df[:-24]


def split_x_test(season):
    '''splitting df to get x test'''
    temp = [x+((season-1)*24) for x in range(1, 25)]
    return feat_df[temp[0]-1:temp[-1]]


def split_x_train(season):
    '''splitting df to get x train'''
    temp = [x+((season-1)*24) for x in range(1, 25)]
    return feat_df.drop(feat_df.index[temp[0]-1:temp[-1]])

# Scaling X train


# Getting episodeCount index and saving it to add back later
ep_index_train = feat_df_x_train.reset_index()
ep_count_train = ep_index_train['episodeCount']


def get_ep_count(df):
    '''saving the episodeCount for df passed in'''
    ep_index_temp = df.reset_index()
    return ep_index_temp['episodeCount']


# Making a copy of the training df with numeric columns only to scale; saving column names
numeric_x_train = feat_df_x_train.drop(['directedBy', 'writtenBy'], axis=1)
col_names_train = numeric_x_train.columns


def numeric_training(season):
    '''returning numeric only training df'''
    return split_x_train(season).drop(['directedBy', 'writtenBy'], axis=1)


def get_col_names(season):
    '''saving column names of training df'''
    return split_x_train(season).columns


# Scaling
a_scaler = preprocessing.StandardScaler()
a_scaler.fit(numeric_x_train)
X_train_scaled_temp = a_scaler.transform(numeric_x_train)

# Adding episodeCount back
X_train_scaled = pd.DataFrame(X_train_scaled_temp, index=ep_count_train)
# adding column names back in
X_train_scaled.columns = col_names_train
X_train_scaled['directedBy'] = feat_df_x_train['directedBy']
X_train_scaled['writtenBy'] = feat_df_x_train['writtenBy']


def scale_train_df(season):
    '''scaling the x train df'''
    a_scaler.fit(numeric_training(season))
    X_train_scaled_temp = a_scaler.transform(numeric_training(season))
    # Adding episodeCount back
    X_training_scaled = pd.DataFrame(
        X_train_scaled_temp, index=get_ep_count(split_x_train(season)))
    # adding column names back in
    # from initial default run; don't need to tailor these (yet)
    X_training_scaled.columns = col_names_train
    X_training_scaled['directedBy'] = split_x_train(season)['directedBy']
    X_training_scaled['writtenBy'] = split_x_train(season)['writtenBy']
    return X_training_scaled

### Repeating process for scaling X test ###


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


def numeric_testing(season):
    return split_x_test(season).drop(['directedBy', 'writtenBy'], axis=1)


def scale_test_df(season):
    # calling function and passing in the x test df to get ep count
    get_ep_count(split_x_test(season))
    a_scaler.fit(numeric_testing(season))
    X_test_scaled_temp = a_scaler.transform(numeric_testing(season))
    # Adding episodeCount back
    X_testing_scaled = pd.DataFrame(
        X_test_scaled_temp, index=get_ep_count(split_x_test(season)))
    # adding column names back in
    # from initial default run; don't need to tailor these (yet)
    X_testing_scaled.columns = col_names_test
    X_testing_scaled['directedBy'] = split_x_test(season)['directedBy']
    X_testing_scaled['writtenBy'] = split_x_test(season)['writtenBy']
    return X_testing_scaled


## the model! ##
automl = AutoML()
automl_settings = {
    "time_budget": 5,
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

# Setting things up ###### MODEL 1 ##### manual
with tab2:

    # in order to try to cut down on memory usage per Streamlit's documentation on resource limits...
    @st.cache
    def load_model():
        '''used to lighten memory load'''
        return pickle.load(f)

    with open("automl.pkl", "rb") as f:
        automl = load_model()
    pred = automl.predict(X_test_scaled)

    rate_compare = season_11_y_test.reindex(
        range(1, 49)).set_index(np.tile(ep_count_test, 2))
    rate_compare['imdbRatings'] = np.append(
        season_11_y_test['imdbRatings'], pred)
    rate_compare['type'] = np.append(
        np.tile('Actual', 24), np.tile('Predicted', 24))
    rate_compare.index.name = 'episodeCount'
    rate_compare = rate_compare.reset_index()

    def prepare_chart_df(season, predicted):
        '''making df of predicted vs. actual y'''
        rate_compare = split_y_test(season).reindex(
            range(1, 49)).set_index(np.tile(get_ep_count(split_y_test(season)), 2))
        rate_compare['imdbRatings'] = np.append(
            split_y_test(season)['imdbRatings'], predicted)
        rate_compare['type'] = np.append(
            np.tile('Actual', 24), np.tile('Predicted', 24))
        rate_compare.index.name = 'episodeCount'
        rate_compare = rate_compare.reset_index()
        return rate_compare

    auto_chart = alt.Chart(rate_compare).mark_bar().encode(
        x=alt.X('type:N', axis=alt.Axis(title='Episode', grid=False)),
        y=alt.Y('imdbRatings:Q', axis=alt.Axis(title='Rating', grid=False)),
        color=alt.condition(
            alt.datum.type == 'Actual',
            alt.value('#fc9272'),
            alt.value('#a6bddb')),
        column=alt.Column('episodeCount:N',
                          header=alt.Header(orient='bottom')),
        tooltip=['imdbRatings']
    ).configure_view(strokeWidth=0).properties(width=50)

    st.subheader(
        "Results of the model's predicted IMDB ratings vs. the actual IMDB ratings.")

    auto_chart

    st.subheader(
        "The non-zero importance of features used by the model to predict IMDB rating.")

    X_feat_imps_info = pd.DataFrame()
    X_feat_imps_info['Features'] = automl.feature_names_in_
    X_feat_imps_info['Importance'] = automl.feature_importances_
    X_feat_imps_info_filtered = X_feat_imps_info[X_feat_imps_info['Importance'] != 0]

    feat_imps_chart = alt.Chart(X_feat_imps_info_filtered).mark_bar().encode(
        x=alt.X('Importance', axis=alt.Axis(
            title='Importance', grid=False)),
        y=alt.Y('Features', axis=alt.Axis(title='Features')),
        color=alt.Color('Features', scale=alt.Scale(scheme='rainbow'), legend=alt.Legend(
            title='Features')),
        tooltip=['Importance']
    ).configure_view(strokeWidth=0).properties(height=500, width=1400)

    feat_imps_chart

    # Getting model stats and info both for best fit and in general

    info_dict = {'Best r2 on validation data:': [1-automl.best_loss],
                 'Training duration of best run:': [automl.best_config_train_time],
                 'R2:': [(1 - sklearn_metric_loss_score('r2', pred, season_11_y_test['imdbRatings']))],
                 'MSE:': [sklearn_metric_loss_score('mse', pred, season_11_y_test['imdbRatings'])],
                 'MAE:': [sklearn_metric_loss_score('mae', pred, season_11_y_test['imdbRatings'])]}

    fit_info = pd.DataFrame(info_dict)
    fit_info.index = ['result']

    col1e, col2e = st.columns(2)

    with col1e:

        st.write("**This was the best-fitting model found**: " +
                 automl.best_estimator + " classifier")

        st.write("**This is the best hyperparameter configuration for the model**:")
        best_config = {k: [v] for k, v in automl.best_config.items()}
        config = pd.DataFrame(best_config)
        config.index = ['result']
        st.dataframe(config)

        st.write("...**and these were the general stats, including the R2 value, which is the proportion of the variation in the dependent variable that is predictable from the independent variable**:")
        st.table(fit_info)
        st.write(
            "_NOTE: Streamlit has limited support and I ran into a bug regarding how to display this table nicely. See my bug report [here](https://github.com/streamlit/streamlit/issues/5828)_")

    st.subheader("Unsurprisingly, the default model doesn't do very well. This is where you get to experiment in modifying the model in order to see if you can come up with a set of features that will create the 'best fit,' meaning you want to try to get a R2 value that is close to 1 (with a positive sign). You can get a head start with this by limiting the selection of features to only those that had a non-zero value, as shown in the plot above.")

# Begin user input part 1 ###### ###### MODEL 2 ##### manual

    imp_feats = list(X_feat_imps_info_filtered['Features'])
    actual_cols = [name.replace('_', ' ') for name in imp_feats]

    col1f, col2f = st.columns(2)

    with col1f:
        feature_choices = st.multiselect(
            'Select the features you would like to include in training a better model:', actual_cols, ['Roz Doyle'], key=1)

    if st.button('Run model!', key=2):
        st.write("Please wait while the model loads... You'll know it's done when the running animation in the corner disappears.")
        choice_X_train = X_train_scaled.loc[:,
                                            X_train_scaled.columns.isin(feature_choices)]
        chosen_ml = AutoML()
        chosen_automl_settings = {
            "time_budget": 5,
            "metric": 'accuracy',
            "task": 'classification',
            "log_file_name": "frasier.log",
        }
        # Train with labeled input data
        chosen_ml.fit(X_train=choice_X_train, y_train=y_train,
                      **chosen_automl_settings)

        @st.cache
        def load_model():
            '''used to lighten memory load'''
            return pickle.load(f)

        with open("automl.pkl", "wb") as f:
            pickle.dump(chosen_ml, f, pickle.HIGHEST_PROTOCOL)

        with open("automl.pkl", "rb") as f:
            chosen_ml = load_model()
        choice_X_test = X_test_scaled.loc[:,
                                          X_test_scaled.columns.isin(feature_choices)]
        choice_pred = chosen_ml.predict(choice_X_test)

        choice_rate_compare = season_11_y_test.reindex(
            list(range(1, 49))).set_index(np.tile(ep_count_test, 2))
        choice_rate_compare['imdbRatings'] = np.append(
            season_11_y_test['imdbRatings'], pred)
        choice_rate_compare['type'] = np.append(
            np.tile('Actual', 24), np.tile('Predicted', 24))
        choice_rate_compare.index.name = 'episodeCount'
        choice_rate_compare = rate_compare.reset_index()

        choice_auto_chart = alt.Chart(choice_rate_compare).mark_bar().encode(
            x=alt.X('type:N', axis=alt.Axis(title='Episode', grid=False)),
            y=alt.Y('imdbRatings:Q', axis=alt.Axis(
                title='Rating', grid=False)),
            color=alt.condition(
                alt.datum.type == 'Actual',
                alt.value('#fc9272'),
                alt.value('#a6bddb')),
            column=alt.Column('episodeCount:N',
                              header=alt.Header(orient='bottom')),
            tooltip=['imdbRatings']
        ).configure_view(strokeWidth=0).properties(width=50)

        st.subheader(
            "Results of the chosen model's predicted IMDB ratings vs. the actual IMDB ratings.")

        choice_auto_chart

        choice_info_dict = {'Best r2 on validation data:': [1-chosen_ml.best_loss],
                            'Training duration of best run:': [chosen_ml.best_config_train_time],
                            'R2:': [(1 - sklearn_metric_loss_score('r2', choice_pred, season_11_y_test['imdbRatings']))],
                            'MSE:': [sklearn_metric_loss_score('mse', choice_pred, season_11_y_test['imdbRatings'])],
                            'MAE:': [sklearn_metric_loss_score('mae', choice_pred, season_11_y_test['imdbRatings'])]}

        choice_fit_info = pd.DataFrame(choice_info_dict)
        choice_fit_info.index = ['result']

        col1g, col2g = st.columns(2)

        with col1g:

            st.write("**This was the best-fitting model found**: " +
                     chosen_ml.best_estimator + " classifier")

            st.write(
                "**This is the best hyperparameter configuration for the model**:")
            choice_best_config = {k: [v]
                                  for k, v in chosen_ml.best_config.items()}
            choice_config = pd.DataFrame(choice_best_config)
            choice_config.index = ['result']
            st.dataframe(choice_config)

            st.write("...**and these were the general stats, including the R2 value, which is the proportion of the variation in the dependent variable that is predictable from the independent variable**:")
            st.table(choice_fit_info)

    else:
        st.write("The model will not run until you click this button.")

# User input part 2 ####### ###### MODEL 3 ##### Functions

    st.subheader("Now, onto grander things... You now can choose on what subset of the data the model gets trained on by selecting which season is dropped. That is, given the 11 seasons, one selected season is set aside for testing and the other 10 are for training.")

    col1hx, col2hx = st.columns(2)
    with col1hx:
        season_to_drop = st.selectbox(
            "Select a season to drop:",
            (range(1, 11)), key=3
        )

    new_y_test = split_y_test(season_to_drop)
    new_y_train = np.array(split_y_train(season_to_drop))
    new_x_test = scale_test_df(season_to_drop)
    new_x_train = scale_train_df(season_to_drop)

    if st.button('Run model!', key=4):
        st.write("Please wait while the model loads... You'll know it's done when the running animation in the corner disappears.")
        new_auto_ml = AutoML()
        new_automl_settings = {
            "time_budget": 5,
            "metric": 'accuracy',
            "task": 'classification',
            "log_file_name": "frasier.log",
        }
        # Train with labeled input data
        new_auto_ml.fit(X_train=new_x_train, y_train=new_y_train,
                        **new_automl_settings)

        @st.cache
        def load_model():
            '''used to lighten memory load'''
            return pickle.load(f)

        with open("automl.pkl", "wb") as f:
            pickle.dump(new_auto_ml, f, pickle.HIGHEST_PROTOCOL)

        with open("automl.pkl", "rb") as f:
            new_auto_ml = load_model()

        new_pred = new_auto_ml.predict(new_x_test)

        new_chart_compare = prepare_chart_df(season_to_drop, new_pred)

        choice_auto_chart = alt.Chart(new_chart_compare).mark_bar().encode(
            x=alt.X('type:N', axis=alt.Axis(title='Episode', grid=False)),
            y=alt.Y('imdbRatings:Q', axis=alt.Axis(
                title='Rating', grid=False)),
            color=alt.condition(
                alt.datum.type == 'Actual',
                alt.value('#fc9272'),
                alt.value('#a6bddb')),
            column=alt.Column('episodeCount:N',
                              header=alt.Header(orient='bottom')),
            tooltip=['imdbRatings']
        ).configure_view(strokeWidth=0).properties(width=50)

        st.subheader(
            "Results of the model's predicted IMDB ratings vs. the actual IMDB ratings.")

        choice_auto_chart

        new_info_dict = {'Best r2 on validation data:': [1-new_auto_ml.best_loss],
                         'Training duration of best run:': [new_auto_ml.best_config_train_time],
                         'R2:': [(1 - sklearn_metric_loss_score('r2', new_pred, new_x_test['imdbRatings']))],
                         'MSE:': [sklearn_metric_loss_score('mse', new_pred, new_x_test['imdbRatings'])],
                         'MAE:': [sklearn_metric_loss_score('mae', new_pred, new_x_test['imdbRatings'])]}

        new_choice_fit_info = pd.DataFrame(new_info_dict)
        new_choice_fit_info.index = ['result']

        col1h, col2h = st.columns(2)

        with col1h:

            st.write("**This was the best-fitting model found**: " +
                     new_auto_ml.best_estimator + " classifier")

            st.write(
                "**This is the best hyperparameter configuration for the model**:")
            new_choice_best_config = {k: [v]
                                      for k, v in new_auto_ml.best_config.items()}
            new_choice_config = pd.DataFrame(new_choice_best_config)
            new_choice_config.index = ['result']
            st.dataframe(new_choice_config)

            st.write("...**and these were the general stats, including the R2 value, which is the proportion of the variation in the dependent variable that is predictable from the independent variable**:")
            st.table(new_choice_fit_info)
    else:
        st.write('The model will not run until you click this button.')

# User changing features and season ##### MODEL 4 ##### Functions
    st.subheader(
        "Now, let's see what happens when you get to control both the season that gets dropped _and_ the features used.")

    col1i, col2i = st.columns(2)
    with col1i:
        another_season_to_drop = st.selectbox(
            "Select a season to drop:",
            (range(1, 11)), key=5
        )
        new_feature_choices = st.multiselect(
            'Select the features you would like to include in training a better model:', actual_cols, ['Roz Doyle'], key=6)

    another_new_y_test = split_y_test(another_season_to_drop)
    another_new_y_train = np.array(split_y_train(another_season_to_drop))
    another_new_x_test = scale_test_df(another_season_to_drop)
    another_new_x_train = scale_train_df(another_season_to_drop)

    if st.button('Run model!', key=7):
        st.write("Please wait while the model loads... You'll know it's done when the running animation in the corner disappears.")
        another_choice_X_train = another_new_x_train.loc[:,
                                                         another_new_x_train.columns.isin(new_feature_choices)]

        another_new_auto_ml = AutoML()
        another_new_automl_settings = {
            "time_budget": 5,
            "metric": 'accuracy',
            "task": 'classification',
            "log_file_name": "frasier.log",
        }
        # Train with labeled input data
        another_new_auto_ml.fit(X_train=another_choice_X_train, y_train=new_y_train,
                                **another_new_automl_settings)

        @st.cache
        def load_model():
            '''used to lighten memory load'''
            return pickle.load(f)

        with open("automl.pkl", "wb") as f:
            pickle.dump(another_new_auto_ml, f, pickle.HIGHEST_PROTOCOL)

        with open("automl.pkl", "rb") as f:
            another_new_auto_ml = load_model()

        another_choice_X_test = another_new_x_test.loc[:,
                                                       another_new_x_test.columns.isin(new_feature_choices)]

        another_new_pred = another_new_auto_ml.predict(another_choice_X_test)

        another_new_chart_compare = prepare_chart_df(
            another_season_to_drop, another_new_pred)

        another_choice_auto_chart = alt.Chart(another_new_chart_compare).mark_bar().encode(
            x=alt.X('type:N', axis=alt.Axis(title='Episode', grid=False)),
            y=alt.Y('imdbRatings:Q', axis=alt.Axis(
                title='Rating', grid=False)),
            color=alt.condition(
                alt.datum.type == 'Actual',
                alt.value('#fc9272'),
                alt.value('#a6bddb')),
            column=alt.Column('episodeCount:N',
                              header=alt.Header(orient='bottom')),
            tooltip=['imdbRatings']
        ).configure_view(strokeWidth=0).properties(width=50)

        st.subheader(
            "Results of the model's predicted IMDB ratings vs. the actual IMDB ratings.")

        another_choice_auto_chart

        another_new_info_dict = {'Best r2 on validation data:': [1-another_new_auto_ml.best_loss],
                                 'Training duration of best run:': [another_new_auto_ml.best_config_train_time],
                                 'R2:': [(1 - sklearn_metric_loss_score('r2', another_new_pred, another_new_y_test['imdbRatings']))],
                                 'MSE:': [sklearn_metric_loss_score('mse', another_new_pred, another_new_y_test['imdbRatings'])],
                                 'MAE:': [sklearn_metric_loss_score('mae', another_new_pred, another_new_y_test['imdbRatings'])]}

        another_new_choice_fit_info = pd.DataFrame(another_new_info_dict)
        another_new_choice_fit_info.index = ['result']

        col1j, col2j = st.columns(2)

        with col1j:

            st.write("**This was the best-fitting model found**: " +
                     another_new_auto_ml.best_estimator + " classifier")

            st.write(
                "**This is the best hyperparameter configuration for the model**:")
            another_new_choice_best_config = {k: [v]
                                              for k, v in another_new_auto_ml.best_config.items()}
            another_new_choice_config = pd.DataFrame(
                another_new_choice_best_config)
            another_new_choice_config.index = ['result']
            st.dataframe(another_new_choice_config)

            st.write("...**and these were the general stats, including the R2 value, which is the proportion of the variation in the dependent variable that is predictable from the independent variable**:")
            st.table(another_new_choice_fit_info)
    else:
        st.write('The model will not run until you click this button.')

#### MISCELLANY AND ACKNOWLEDGEMENTS TAB ####

with tab3:

    st.write("Did you know that there exists a compilation (multiple, in fact!) of every time Frasier says 'Oh, dear God!' throughout the show? Well, now you do...")

    video_file = open('oh dear god.mp4', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

    st.caption(
        "From the YouTube channel of [ParaFrasier] (https://www.youtube.com/@parafrasier).")

    st.write(
        "Here are the words and phrases from Frasier in my daily parlance (all with the correct intonations, of course): \n * 'What fresh hell is this?' \n * 'Rrrrrromping with my schoolchums...' \n * 'If less is more, think how much more more will be!' \n * 'I'm delightfully quirky.' \n *  'I was punched in the face by a man now dead!' \n *  'Her lips said no but her eyes said read my lips.' \n *  'I am wounded!' \n * 'But if a cricket's brain is _this_ big, think how big a gecko's brain is!' \n * 'Cute but stupid.' \n * 'Snausages!' \n * 'Copernicus called and you are not the centre of the universe.' \n * 'That dingo's got your baby.'")

#### MISCELLANY AND ACKNOWLEDGEMENTS TAB ####

with tab4:

    st.subheader("Some general comments.")

    st.write("I use FLAML, Microsoft's 'Fast Library for Automated Machine Learning & Tuning.' Several primary reasons for this choice are as follows. \n 1) The data is limited. Currently, I have not utilised the aspect of this data that captures every single word ever said in the show. Instead, I have 'collapsed' things down by episode. This means that, instead of approximately 775,000 observations, there are only about 260 observations. This 'feature engineering' results in a very small number that is far from ideal for machine learning purposes. However, it's what I've got for now (until I can learn more about how to handle the much larger, original dataset). \n 2) I set out to make the ML part of the web app more focused on user interaction as opposed to showing all the nitty-gritty details of what's going on under the machine learning hood. You can see a lot of that nitty-gritty stuff in my code, but FLAML makes this user-friendly goal much more easily achievable in a limited time frame. Because ain't nobody got time to manually tune dozens of hyperparameters... And, honestly, this is an app I'd like to eventually disseminate publically, so it seems silly to go through the bother of trying to get pyplot or somesuch to work with Streamlit just to throw up a bunch of graphics about stuff like confusion matrices and the like. But I do utilise underlying machine learning results and such by displaying the models' stats. \n 3) As you may or may not know, there are plans for a _Frasier_ reboot-continuation. Personal opinions and trepidations aside, I believe that this will prove a fascinating case study to see how the original _Frasier_ audience—of the '90's and '00's—interacts with a revival made 20 years later (as opposed to the newer audience). This is a balancing act we are seeing more and more in our media-saturated world as creators want to capture the widest audience possible. Creators have to decide how much to cater to the original audience while also catering to the new audiences. It's a snarly mess of social forces as generational ideologies continually shift and self-fashioning becomes ubiquitous. \n All that goes to say, I'm endlessly curious about how a _Frasier_ revival will be received. My project here represents a meagre attempt to become part of the larger discourse to see what (if anything) can be learned via data science techniques like machine learning.")

    st.subheader(
        "Sundry items like acknowledgements, links, etc. and so forth.")

    st.write(
        "You can find the GitHub folder [here](https://github.com/amrungwaew/frasier_project).")

    st.write(
        "My thanks to Chip Oglesby, from whom the nice, tidy data was [sourced](https://data.world/chipoglesby/frasier-crane-television-transcripts).")

    st.subheader("Questions and quandries I'm still working through")

    st.write("I'm sure my thought process and execution is far from perfect, this being my first attempt to use machine learning. Here are some of the outstanding things I've been rolling around in my mind, but didn't have the time/domain knowledge to actually address in my code.")

    st.write("1. I originally began making functions to make a pipeline. However, it was very late in the process and my data had not been organised in such a way that a pipeline could easily be imposed. But the functions still came in handy. (I didn't group them because I'm scared of breaking things if something ends up out of order for the stuff outside of the functions in the manual parts.) (Season 8, episode 11) \n 2. I don't know if the important features chance from model to model. For now I'm just using the important features gathered from the first instance of AutoML(). But I may want to do some tweaking at some point to see if the features do in fact chance from instance to instance. And if they do, I want to understand why... and I don't know enough to ask that yet. (Season 7, episode 13) \n 3. Streamlit is very nice for simply stuff but very buggy and tedious for others. Other than the reported bug instance, there were a couple other times I ran into bugs. Also, I'm regularly going over the Streamlit resources (mainly memory, I think) so it keeps crashing and I have to keep rebooting. I submitted a form to see if they would allot me extra resources since this is a school project but alas, crickets. (Season 9, episode 2) \n 4. I ran out of time for now, but I want to be able to add a feature of a table or metric that can keep track of a user's 'score' with trying out different models. (Season 9, episode 14) \n 5. I don't know how to get around Streamlit constantly re-running when changes happen. I decided to leave all the model training inside the if loops to try to limit the amount of clogging up, but everything is still just slow and messy. (Season 8, episode 18)")
