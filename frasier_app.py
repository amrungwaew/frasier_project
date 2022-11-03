## Anna Jeffries, October 2022

from statistics import mean
import pandas as pd 
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

st.set_page_config(
    page_title="CMSE 830 Midterm Project",
    layout='wide' # configuring layout to be wide (as opposed to centred) to to take up more screen space
)

st.header("Welcome to Anna's web app project that lets you look at information extracted from the transcripts of the iconic American TV show, *Frasier*.")

st.subheader("Here, you can view information by season as well as by episode.")

# columns for spacing input widgets
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8) # columns for button spacing

with col1:
    season_select = st.selectbox(
            "Select a season:",
            (range(1,12))
    )

with col5:
    episode_select = st.selectbox(
            "Which episode of season " + str(season_select) + "?",
            (range(1,25))
    )

df_frasier = pd.read_csv('tidyTranscripts.csv')

# word count for every episode orgd by season
df_frasier_totalwords = df_frasier.groupby([
    'season','episode','characterName','gender', 'actorName','characterType','episodeCount',
    'originalAirDate','title','viewershipInMillions','imdbRatings']).size().reset_index(name='total_words')

# word count organized by character for every episode orgd by season
df_frasier_characterwords = df_frasier.groupby([
    'season','episode','characterName','gender', 'actorName','characterType','episodeCount',
    'originalAirDate','title','viewershipInMillions','imdbRatings']).size().reset_index(name='total_words')

# functions to get the season and episode info
def get_season(season):
    '''selecting the entries matching the season selected'''
    return df_frasier_characterwords.where(df_frasier_characterwords['season'] == season).dropna()

def get_season_words():
    '''this is simply summing the words in a given season and returning as list'''
    sums = []
    for i in range(1,12):
        ss = sum(df_frasier_characterwords.where(df_frasier_characterwords['season'] == i).dropna()['total_words'])
        sums.append(ss)
    return sums

def get_episode(episode, the_season):
    '''takes 2 parameters because the season selected needs to be used to select the right episode'''
    return the_season.where(the_season['episode'] == episode).dropna()

# calling functions to create the df needed for plotting
df_season = get_season(season_select)
df_episode = get_episode(episode_select, df_season)


# getting the list of season word totals as well as a list of just the seasons and adding them to 
# the main DataFrame called in season_plot because Altair is freaking annoying and refuses to take
# in anything other than the one dataset so I'm having to add awkward hanging columns to get the
# information into the chart that I need to have....
season_words = get_season_words()
season_totals = pd.DataFrame({'totals':season_words})
df_frasier_totalwords['season_total'] = season_totals
df_frasier_totalwords['seasonlist'] = pd.DataFrame(range(1,12))

rating_avg = []
for i in range(1,12):
    rating_avg.append(mean(df_frasier_totalwords['imdbRatings'].where(df_frasier_totalwords['season'] == i).dropna()))

viewing_avg = []
for i in range(1,12):
    viewing_avg.append(mean(df_frasier_totalwords['viewershipInMillions'].where(df_frasier_totalwords['season'] == i).dropna()))



df_frasier_totalwords['rating_avg'] = pd.Series([round(num,2) for num in rating_avg])
df_frasier_totalwords['viewing_avg'] = pd.Series([round(num,2) for num in viewing_avg])


df_frasier_characterwords['rating_avg'] = pd.Series([round(num,2) for num in rating_avg])
df_frasier_characterwords['viewing_avg'] = pd.Series([round(num,2) for num in viewing_avg])

# plotting the season chart with total words by season
season_plot = alt.Chart(df_frasier_totalwords,padding={'left': 0, 'top': 25, 'right': 0, 'bottom': 5}).mark_bar(size=30).encode(
    x=alt.X('seasonlist', axis=alt.Axis(title='Season',grid=False)),
    y=alt.Y('season_total',axis=alt.Axis(title='Total number of words')),
    color=alt.condition(
        alt.datum.seasonlist == season_select,  
        alt.value('crimson'),    
        alt.value('darkgrey')),
    tooltip=['season_total'] 
    ).configure_view(strokeWidth=0).properties(width=600).interactive()

# plotting the episode chart with words by character
episode_plot = alt.Chart(df_episode,padding={'left': 0, 'top': 25, 'right': 5, 'bottom': 5}
    ).mark_bar(size=35).encode(
    x=alt.X('characterName', axis=alt.Axis(title='Characters')),
    y=alt.Y('total_words', axis=alt.Axis(title='Total number of words')),
    color=alt.Color('characterName',
        scale=alt.Scale(scheme='turbo'), legend=alt.Legend(title='Characters', orient='right')),
    tooltip=['total_words','actorName','characterType','gender'] 
    ).configure_view(strokeWidth=0).properties(width=alt.Step(50)).interactive()

# columns for spacing plots
col11, col22 = st.columns(2)

with col11:
    season_plot
with col22:
    episode_plot

## SECOND SET OF PLOTS

st.subheader("Here, you can view information by character in a given season.")

main_ch = df_frasier_characterwords.where(df_frasier_characterwords['characterType'] == 'main').dropna()
main_ch_names = list(main_ch['characterName'].unique())
recur_ch = df_frasier_characterwords.where(df_frasier_characterwords['characterType'] == 'recurring').dropna()
recur_ch_names = list(recur_ch['characterName'].unique())

with st.container():

    col1a, col2a, col3a, col4a, col5a, col6a, col7a, col8a = st.columns(8)

    with col1a:
        ch_select = st.selectbox(
                "Select a main character:",
                (main_ch_names)
        )

    with col2a:
        ch_season_select = st.selectbox(
                "Which season?",
                (range(1,12))
        )


    def get_ch_season(ch_season, char):
        '''selecting the entries matching the character name and season selected'''
        df = df_frasier_characterwords.loc[df_frasier_characterwords['season'] == ch_season]
        return df.loc[df['characterName'] == char]


    # calling functions to create the df needed for plotting
    df_ch_season = get_ch_season(ch_season_select, ch_select)

    col11a, col22a, col33a = st.columns(3)

    ch_season_plot = alt.Chart(df_ch_season,padding={'left': 0, 'top': 25, 'right': 0, 'bottom': 5}).mark_line(
        color='gold',point=alt.OverlayMarkDef(color="white",size=80),width=15).encode(
        x=alt.X('episode', axis=alt.Axis(title='Episodes',grid=False)),
        y=alt.Y('total_words',axis=alt.Axis(title='Total number of words')),
        tooltip=['title','total_words','actorName','gender'] 
        ).configure_view(strokeWidth=0).properties(width=450).interactive()

    with col11a:
        ch_season_plot


    with col22a:
        achoice = st.checkbox('I would like to compare ' + ch_select + ' with another main character')

        if achoice:
            
            ch1_select = st.selectbox(
                "Select a main character:", key='secondarymain', options=main_ch_names)
                
            df_ch1_season = get_ch_season(ch_season_select, ch1_select)
            df_ch1_season_combo = pd.concat([df_ch_season, df_ch1_season], ignore_index=True)

            ch_season_combo_plot = alt.Chart(df_ch1_season_combo,padding={'left': 0, 'top': 25, 'right': 0, 'bottom': 5}).mark_line(
                    point=alt.OverlayMarkDef(size=50),width=10).encode(
                    x=alt.X('episode', axis=alt.Axis(title='Episodes',grid=False)),
                    y=alt.Y('total_words',axis=alt.Axis(title='Total number of words')),
                    color=alt.Color('characterName',scale=alt.Scale(scheme='set2'),
                    legend=alt.Legend(title='Characters', orient='bottom')),
                    tooltip=['title','total_words','actorName','gender']
                    ).configure_view(strokeWidth=0).properties(width=450).interactive()
                
            ch_season_combo_plot

    with col33a:
        anotherchoice = st.checkbox('I would like to compare ' + ch_select + ' with a recurring character',key='recurring')

        if anotherchoice:
            
            ch_recur_select = st.selectbox(
                "Select a recurring character:", key='secondaryrecur', options=recur_ch_names)
            
            df_ch_rec_season = get_ch_season(ch_season_select, ch_recur_select)
            df_chrec_season_combo = pd.concat([df_ch_season, df_ch_rec_season], ignore_index=True)

            df_chrec_season_combo_plot = alt.Chart(df_chrec_season_combo,padding={'left': 0, 'top': 25, 'right': 0, 'bottom': 5}).mark_line(
                point=alt.OverlayMarkDef(size=50),width=10).encode(
                x=alt.X('episode', axis=alt.Axis(title='Episodes',grid=False)),
                y=alt.Y('total_words',axis=alt.Axis(title='Total number of words')),
                color=alt.Color('characterName',scale=alt.Scale(scheme='rainbow'),
                legend=alt.Legend(title='Characters', orient='bottom')),
                tooltip=['title','total_words','actorName','gender']
                ).configure_view(strokeWidth=0).properties(width=450).interactive()
            
            df_chrec_season_combo_plot


# THIRD SET

st.subheader("*** Everything below is WIP ***")

with st.container():

    col1aa, col2aa, col3aa, col4aa = st.columns(4)

    with col1aa:
        ch_options = st.multiselect('Select as many characters as you please',
        main_ch_names + recur_ch_names,['Frasier Crane'])
    
    def get_ch_show(char):
        '''selecting the entries matching the character name'''
        return df_frasier_characterwords.loc[df_frasier_characterwords['characterName'] == char]
    

    df_selection_show = pd.DataFrame()
    for person in ch_options:
        df_selection_show = pd.concat([df_selection_show, get_ch_show(person)])

    # seasons_rect = pd.DataFrame({
    #     'start': [0,24,48,72,96,120,144,168,192,216,240,264],
    #     'stop': [24,48,72,96,120,144,168,192,216,240,264,288]
    #     })

    ch_show_plot = alt.Chart(df_selection_show).mark_line().encode(
        x=alt.X('episodeCount', axis=alt.Axis(title='Episodes',grid=False)),
        y=alt.Y('total_words',axis=alt.Axis(title='Total number of words')),
        color=alt.Color('characterName',scale=alt.Scale(scheme='set2'),
        legend=alt.Legend(title='Characters', orient='bottom'))).properties(height=500,width=1400).interactive()

    # areas = alt.Chart(seasons_rect).mark_rect(opacity=0.1).encode(
    #         x='start', x2='stop',
    #         color=alt.Color('stop',scale=alt.Scale(scheme='rainbow'),
    #         legend=alt.Legend(title='Seasons', orient='bottom')))

    ch_show_plot

    # with col1aa:
    #     gender_select = st.checkbox("I would like to view across-show statistics categorically",key='cat')

    #     if gender_select:
    #         cat = st.selectbox("Select a category:", key='cats', options=['Gender',''])
    
