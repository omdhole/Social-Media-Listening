import streamlit as st
import requests
import openai
import json
import plotly.express as px
import requests
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from youtube_search import YoutubeSearch
from youtube_comment_downloader import YoutubeCommentDownloader

# Use markdown + HTML for colored title - Added
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
# st.markdown(
#     "<h1 style='color: #1D1D1D; text-align: center;'>🎬 Sentiment Dashboard</h1>",
#     unsafe_allow_html=True
# )
st.markdown(
    """
    <div style='background-color: black; padding: 10px; border-radius: 8px; text-align: center;'>
        <h1 style='color: white; font-weight: bold; margin: 0;'>🎬 Sentiment Dashboard</h1>
    </div>
    <div style='margin-top: 30px;'></div>

    """,
    unsafe_allow_html=True
)

OMDB_API_KEY = st.secrets["api_keys"]["OMDB_API_KEY"]
#LLM_TOKEN = st.secrets["api_keys"]["LLM_TOKEN"]
LLM_ENDPOINT = "https://models.github.ai/inference"
MODEL_NAME = "openai/gpt-4.1"

def fetch_omdb_data(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()
    #st.write(data)
    if data.get("Response") == "True":
        return data
    else:
        return None
    

# KPI colour change - Added
def uniform_metric(label, value, color="#404041"):
    st.markdown(f"""
        <div style='
            background-color: #F0F2F6;
            padding: 16px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 2px 2px 4px rgba(0,0,0,0.05);
            '>
            <div style='color: {color}; font-weight: bold; font-size: 18px;'>{label}</div>
            <div style='font-size: 24px;'>{value}</div>
        </div>
    """, unsafe_allow_html=True)

# --- Main UI ---

title_input = st.text_input("🔎 Enter movie title")

# Use session state to store if movie was searched and the movie title
if 'movie_searched' not in st.session_state:
    st.session_state.movie_searched = False
if 'movie_title' not in st.session_state:
    st.session_state.movie_title = ""
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'overview'

search_clicked = st.button("Search")

if search_clicked:
    if not title_input.strip():
        st.warning("Please enter title.")
    else:
        st.session_state.movie_searched = True
        st.session_state.movie_title = title_input.strip()
        st.session_state.selected_tab = 'overview'  # reset to overview on new search

if st.session_state.movie_searched:
    # Fetch data once here using st.session_state.movie_title
    omdb_data = fetch_omdb_data(st.session_state.movie_title)
    if omdb_data is None:
        st.error("Movie not found in OMDb. Please try another title.")
    
    else:
        st.markdown("---")
        tab1, tab2 = st.tabs([":material/dashboard: Overview & Performance" , ":material/insights: Social Media Insights"])

        st.markdown(
            """
            <style>
            /* Target the tabs container */
            .css-1d391kg {  /* This class might change; inspect your app to confirm */
                justify-content: flex-start !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
        with tab1:
            # --- Top metrics line ---
            col1, col2 = st.columns(2)
            with col1:
                uniform_metric("IMDb Rating", omdb_data.get("imdbRating", "N/A"))
            with col2:
                uniform_metric("Runtime (min)", omdb_data.get("Runtime", "N/A"))

            st.markdown("---")

            # --- Engagement, Demographics, Sentiment Clustering ---
            st.markdown(
                """
                <div style="background-color: #2F2F30; padding: 2px; border-radius: 5px ; text-align: center;">
                    <h5 style="color: white;"> ⓘ Overview</h5>
                </div>
                <div style="height: 20px;"></div>  <!-- Spacer with 20px height -->
                """,
                unsafe_allow_html=True    
            )

            col1, col2 = st.columns(2)

            with col1:                
                st.markdown(f"**Release Date:** {omdb_data.get('Released', 'N/A')}")
                st.markdown(f"**Genre:** {omdb_data.get('Genre', 'N/A')}")
                st.markdown(f"**Season:** {omdb_data.get('totalSeasons', 'N/A')}")
                st.markdown(f"**Writers:** {omdb_data.get('Writer', 'N/A')}")
                st.markdown(f"**Director:** {omdb_data.get('Director', 'N/A')}")
                st.markdown(f"**Actors:** {omdb_data.get('Actors', 'N/A')}")
                st.markdown(f"**Language:** {omdb_data.get('Language', 'N/A')}")
                st.markdown(f"**Awards:** {omdb_data.get('Awards', 'N/A')}")
                st.markdown(f"**Plot:** {omdb_data.get('Plot', 'N/A')}")

            with col2:
                image_link = omdb_data.get('Poster', 'N/A')
                if image_link != 'N/A':
                    st.markdown(
                        f"""
                        <div style='text-align: center;'>
                            <img src="{image_link}" width="300" />
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.write("No image available")
                

                # # Display the image with a caption
                # st.image(image_link, width=300)

            st.markdown("---")
            st.markdown(
                """
                <div style="background-color: #2F2F30; padding: 2px; border-radius: 5px ; text-align: center;">
                    <h5 style="color: white;"> 📝 Score Card</h5>
                </div>
                <div style="height: 20px;"></div>  <!-- Spacer with 20px height -->
                """,
                unsafe_allow_html=True    
            )

            Ratings = omdb_data.get("Ratings",[])

            if Ratings:
                df_score = pd.DataFrame(Ratings)
                df_score["Value"] = df_score["Value"].apply(lambda x: x.strip())
                df_score = df_score.reset_index(drop=True)
                st.dataframe(df_score, use_container_width=True)
            else:
                st.info("Score card data unavailable.")

            st.markdown("---")
            
            def get_video_url(movie_name):
                results = YoutubeSearch(movie_name + " official trailer", max_results=1).to_dict()
                if results:
                    video_id = results[0]['id']
                    return video_id,f"https://www.youtube.com/watch?v={video_id}"
                else:
                    return 0,"No video found."

            video_id, video_url=get_video_url(title_input)

            # Replace with your API key
            API_KEY_YT = st.secrets["api_keys"]["API_KEY_YT"]

            # Initialize YouTube API client
            youtube = build('youtube', 'v3', developerKey=API_KEY_YT)

            def get_video_stats(video_id):
                # Get video statistics
                video_response = youtube.videos().list(
                    part='statistics,snippet',
                    id=video_id
                ).execute()

                if not video_response['items']:
                    return "Video not found."

                video_data = video_response['items'][0]
                stats = video_data['statistics']
                snippet = video_data['snippet']

                # Get channel statistics
                channel_id = snippet['channelId']
                channel_response = youtube.channels().list(
                    part='statistics',
                    id=channel_id
                ).execute()

                channel_stats = channel_response['items'][0]['statistics']

                return {
                    'title': snippet['title'],
                    'views': stats.get('viewCount', 'N/A'),
                    'likes': stats.get('likeCount', 'N/A'),
                    'comments': stats.get('commentCount', 'N/A'),
                    'subscribers': channel_stats.get('subscriberCount', 'N/A'),
                    'channel': snippet['channelTitle'],
                    'video_url': f"https://www.youtube.com/watch?v={video_id}"
                }

            # Example usage
            #video_id = 'Wk5OxqtpBR4'  # Replace with actual video ID
            video_data = get_video_stats(video_id)

            # --- Video Card ---
            st.markdown(
                f"""
                <div style="
                    background-color: #1F1F23;
                    padding: 20px;
                    border-radius: 15px;
                    color: #F5F5F5;
                    font-family: Arial, sans-serif;
                ">
                    <h3 style='margin-bottom:5px;'>🎬 <a href="{video_data['video_url']}" target="_blank" style="color:#F5F5F5; text-decoration:none;">{video_data['title']}</a></h3>
                    <p style='margin:2px 0; font-size:14px;'>📺 Channel: <strong>{video_data['channel']}</strong> | 👥 Subscribers: <strong>{f"{int(video_data['subscribers']):,}"}</strong></p>
                    <div style="display:flex; gap:20px; margin-top:10px;">
                        <div style="text-align:center; flex:1; background-color:#2F2F30; padding:10px; border-radius:10px;">
                            👁️ <br><strong>{f"{int(video_data['views']):,}"}</strong> Views
                        </div>
                        <div style="text-align:center; flex:1; background-color:#2F2F30; padding:10px; border-radius:10px;">
                            👍 <br><strong>{f"{int(video_data['likes']):,}"}</strong> Likes
                        </div>
                        <div style="text-align:center; flex:1; background-color:#2F2F30; padding:10px; border-radius:10px;">
                            💬 <br><strong>{f"{int(video_data['comments']):,}"}</strong> Comments
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with tab2:
        
            ###################### YOU TUBE FUNCTIONALITY  #######################
            downloader = YoutubeCommentDownloader()
            comments_generator = downloader.get_comments_from_url(video_url)

            comments_data = []
            limit = 200
            count = 0

            for comment in comments_generator:
                comment_info = {
                    'text': comment.get('text', ''),
                    'author': comment.get('author', ''),
                    'time': comment.get('time', ''),
                    'likes': comment.get('likes', 0),
                    'replyCount': comment.get('replyCount', 0),
                    'channelId': comment.get('channelId', '')
                }
                comments_data.append(comment_info)
                count += 1
                if count >= limit:
                    break

            # print(comments_data)

            def format_comments_for_prompt(comments_data):
                formatted = ""
                for i, comment in enumerate(comments_data, 1):
                    text = comment.get('text', '').replace('\n', ' ')
                    author = comment.get('author', 'Unknown')
                    formatted += f"{i}. {text} (by {author})\n"
                return formatted

            comments_text = format_comments_for_prompt(comments_data)

            ### Feeding data to llm
            LLM_TOKEN_YOUTUBE = st.secrets["api_keys"]["LLM_TOKEN_YOUTUBE"]
            LLM_ENDPOINT = "https://models.github.ai/inference"
            MODEL_NAME = "openai/gpt-4.1"

            def get_movie_youtube_comments_summary(comments_text):
                prompt = f"""
                You are a movie critic and data analyst. Analyze the following YouTube comments for the movie: {comments_text}

                Given the youtube comments, generate the following metrics in JSON format:

                1. "Sentiment Analysis": 
                    a. "Public Opinion": provide one concise bullet point each for positive, negative, and neutral sentiment.
                    b. "Emotional Intensity": provide one concise bullet point each for key emotions such as love, disappointment, and anger.

                2. "Themes":
                    Provide 20 single-word topics with weights indicating their prominence in the movie comments.  
                    Example format → {{"Acting": 25, "Direction": 20, "Music": 15, "Plot": 18, "Complexity": 10}}

                3. "Audience Preferences": One bullet point each summarising comments on genre, cast, director
 
                4. "Expectations vs. Reality": One bullet point summarising viewers expectations based on the trailer and how well the movie met those expectations in two lines.
 
                5. "Memorable Quotes": Identify the most talked-about scenes from the comments and describe in two lines.
                
                6. "Criticism": Spot specific viewer complaints—like pacing issues, plot holes, or weak performances—to understand what didn’t resonate in the trailer (in bullet points)
        
                7. "Viewer engagement": Does viewer mention wanting to see sequels or next season (use bullet points)
 
                8. "Cultural Insights":
                    "Cultural References & Values": Cultural references or values that comments identify with or react against (in one bullet point)
                    "Social Issues & Generational Perspectives": How the comments reflect or engage with social issues, identity, or generational perspectives (in 3 bullet points)
                    "Emotional Tone & Viewer Mindset": The emotional tone and viewer mindset, including specific triggers that evoke strong reactions—such as excitement, nostalgia, discomfort, or curiosity (in 3 bullet points)
                        
                9. "Production Review": Comments on visuals, sound, effects, or direction quality. Provide exactly 3 concise bullet points
                
                10. "Narrative Structure & Plot Complexity": Remarks on pacing, clarity, twists, or ending preferences. Provide exactly 3 concise bullet points
                
                11. "Aesthetics": Opinions on costume, set design, color, or overall aesthetics. Provide exactly 3 concise bullet points
        
                Provide output strictly as valid JSON only with these keys exactly.
                """

                messages = [
                    {"role": "system", "content": "You are a movie critic and data analyst."},
                    {"role": "user", "content": prompt}
                ]

                client = ChatCompletionsClient(
                    endpoint=LLM_ENDPOINT,
                    credential=AzureKeyCredential(LLM_TOKEN_YOUTUBE)
                )

                response = client.complete(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    max_tokens=20000
                )
                
                content = response.choices[0].message.content.strip()
                # st.write("LLM raw output:", content)
                metrics = json.loads(content)
                return metrics
            
                ###################### REDDIT FUNCTIONALITY  ######################################################################################

            def fetch_reddit_posts(title_input, limit=15):
                url = f"https://www.reddit.com/search.json?q={title_input}&limit={limit}"
                headers = {'User-agent': 'MovieBot 0.1'}
                response = requests.get(url, headers=headers)
                
                try:
                    data = response.json()
                    posts_data = [
                        {
                            "title": post["data"].get("title", ""),
                            "selftext": post["data"].get("selftext", ""),
                            "author": post["data"].get("author", ""),
                            "subreddit": post["data"].get("subreddit", ""),
                            "permalink": f"https://www.reddit.com{post['data'].get('permalink', '')}",
                            "score": post["data"].get("score", 0),
                            "num_comments": post["data"].get("num_comments", 0)
                        }
                        for post in data["data"]["children"]
                    ]
                    
                    return posts_data
                
                except ValueError:
                    return []

            def format_posts_for_prompt(posts_data):
                formatted = ""
                for i, post in enumerate(posts_data, 1):
                    title = post.get('title', '').replace('\n', ' ').strip()
                    selftext = post.get('selftext', '').replace('\n', ' ').strip()
                    author = post.get('author', 'Unknown')
                    subreddit = post.get('subreddit', 'Unknown')
                    permalink = post.get('permalink', '')
                    score = post.get('score', 0)
                    num_comments = post.get('num_comments', 0)

                    content = title
                    if selftext:
                        content += " - " + selftext

                    formatted += (
                        f"{i}. {content} "
                        f"(Author: {author}, Subreddit: {subreddit}, Score: {score}, Comments: {num_comments})\n"
                        f"Link: {permalink}\n\n"
                    )
                return formatted
            
            posts = fetch_reddit_posts(title_input)
            formatted_text = format_posts_for_prompt(posts)

            ### Feeding data to llm
            LLM_TOKEN_REDDIT = st.secrets["api_keys"]["LLM_TOKEN_REDDIT"]
            LLM_ENDPOINT = "https://models.github.ai/inference"
            MODEL_NAME = "openai/gpt-4.1"

            def get_movie_reddit_posts_summary(formatted_text):
                prompt = f"""
                You are a movie critic and data analyst. Analyze the following reddit posts for the movie: {formatted_text}

                Given the youtube comments, generate the following metrics in JSON format:

                1. "Sentiment Analysis": 
                    a. "Public Opinion": provide one concise bullet point each for positive, negative, and neutral sentiment.
                    b. "Emotional Intensity": provide one concise bullet point each for key emotions such as love, disappointment, and anger.
                2. "Themes":
                    Provide 20 single-word topics with weights indicating their prominence in the movie comments.  
                    Example format → {{"Acting": 25, "Direction": 20, "Music": 15, "Plot": 18, "Complexity": 10}}
                3. "Audience Preferences": One bullet point each summarising comments on genre, cast, director
                4. "Expectations vs. Reality": One bullet point summarising viewers expectations based on the trailer and how well the movie met those expectations in two lines.
                5. "Memorable Quotes": Identify the most talked-about scenes from the comments and describe in two lines.        
                6. "Criticism": Spot specific viewer complaints—like pacing issues, plot holes, or weak performances—to understand what didn’t resonate in the trailer (in bullet points)
                7. "Viewer engagement": Does viewer mention wanting to see sequels or next season (use bullet points)
                8. "Cultural Insights":
                    "Cultural References & Values": Cultural references or values that comments identify with or react against (in one bullet point)
                    "Social Issues & Generational Perspectives": How the comments reflect or engage with social issues, identity, or generational perspectives (in 3 bullet points)
                    "Emotional Tone & Viewer Mindset": The emotional tone and viewer mindset, including specific triggers that evoke strong reactions—such as excitement, nostalgia, discomfort, or curiosity (in 3 bullet points)       
                9. "Production Review": Comments on visuals, sound, effects, or direction quality. Provide exactly 3 concise bullet points
                10. "Narrative Structure & Plot Complexity": Remarks on pacing, clarity, twists, or ending preferences. Provide exactly 3 concise bullet points
                11. "Aesthetics": Opinions on costume, set design, color, or overall aesthetics. Provide exactly 3 concise bullet points
                Provide output strictly as valid JSON only with these keys exactly.
                """
                messages = [
                    {"role": "system", "content": "You are a movie critic and data analyst."},
                    {"role": "user", "content": prompt}
                ]

                client = ChatCompletionsClient(
                    endpoint=LLM_ENDPOINT,
                    credential=AzureKeyCredential(LLM_TOKEN_REDDIT)
                )

                response = client.complete(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    max_tokens=20000
                ) 
                content_red = response.choices[0].message.content.strip()
                #st.write("LLM raw output REDDIT:", content_red)
                metrics_red = json.loads(content_red)
                return metrics_red

            ###################### GOOGLE PLAY MOVIES FUNCTIONALITY  #########################

            ### Feeding data to llm
            LLM_TOKEN_GOOGLE = st.secrets["api_keys"]["LLM_TOKEN_GOOGLE"]
            LLM_ENDPOINT = "https://models.github.ai/inference"
            MODEL_NAME = "openai/gpt-4.1"

            def get_movie_google_play_reviews_summary(title_input: str) -> str:
                prompt = f""" Please provide the following data for [specific topic or entity], including the source of the information for each data point
                You are a movie analyst. For the movie '{title_input}', provide:

                Please provide:
                1. "Sentiment Analysis": 
                    a. "Public Opinion": provide one concise bullet point each for positive, negative, and neutral sentiment.
                    b. "Emotional Intensity": provide one concise bullet point each for key emotions such as love, disappointment, and anger.
                2. "Themes":
                    Provide 20 single-word topics with weights indicating their prominence in the movie comments.  
                    Example format → {{"Acting": 25, "Direction": 20, "Music": 15, "Plot": 18, "Complexity": 10}}
                3. "Audience Preferences": One bullet point each summarising comments on genre, cast, director
                4. "Expectations vs. Reality": One bullet point summarising viewers expectations based on the trailer and how well the movie met those expectations in two lines.
                5. "Memorable Quotes": Identify the most talked-about scenes from the comments and describe in two lines.        
                6. "Criticism": Spot specific viewer complaints—like pacing issues, plot holes, or weak performances—to understand what didn’t resonate in the trailer (in bullet points)
                7. "Viewer engagement": Does viewer mention wanting to see sequels or next season (use bullet points)
                8. "Cultural Insights":
                    "Cultural References & Values": Cultural references or values that comments identify with or react against (in one bullet point)
                    "Social Issues & Generational Perspectives": How the comments reflect or engage with social issues, identity, or generational perspectives (in 3 bullet points)
                    "Emotional Tone & Viewer Mindset": The emotional tone and viewer mindset, including specific triggers that evoke strong reactions—such as excitement, nostalgia, discomfort, or curiosity (in 3 bullet points)       
                9. "Production Review": Comments on visuals, sound, effects, or direction quality. Provide exactly 3 concise bullet points
                10. "Narrative Structure & Plot Complexity": Remarks on pacing, clarity, twists, or ending preferences. Provide exactly 3 concise bullet points
                11. "Aesthetics": Opinions on costume, set design, color, or overall aesthetics. Provide exactly 3 concise bullet points

                Note: Refer only google play movies data no other data source is accepted
                Provide output strictly as valid JSON only with these keys exactly.
                """
                messages = [
                    {"role": "system", "content": "You are a movie critic and data analyst."},
                    {"role": "user", "content": prompt}
                ]

                client = ChatCompletionsClient(
                    endpoint=LLM_ENDPOINT,
                    credential=AzureKeyCredential(LLM_TOKEN_GOOGLE)
                )

                response = client.complete(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    max_tokens=20000
                ) 
                content_google = response.choices[0].message.content.strip()
                #st.write("LLM raw output GOOGLE:", content_google)
                metrics_google = json.loads(content_google)
                return metrics_google


            ###################### END FUNCTIONALITY  ##################################################################
            st.markdown(
                """
                <div style='background-color: black; padding: 10px; border-radius: 8px; text-align: center;'>
                    <h4 style='color: white; font-weight: bold; margin: 0;'>📊 Audience Insights & Analysis</h4>
                </div>
                <div style='margin-top: 30px;'></div>

                """,
                unsafe_allow_html=True
            )
            if comments_text is None:
                st.error("comments not found in youtube. Please try another title.")
            
            else:


                # Fetch LLM metrics live
                with st.spinner("Fetching Social Media Insights from LLM..."):
                    llm_metrics = get_movie_youtube_comments_summary(comments_text)
                    llm_metrics_red = get_movie_reddit_posts_summary(formatted_text)
                    llm_metrics_google = get_movie_google_play_reviews_summary(title_input.strip())

                    st.markdown("""
                <style>
                /* 🎨 Clean, Modern Tab Design */

                /* Tab container layout */
                .stTabs [data-baseweb="tab-list"] {
                    gap: 10px;
                    justify-content: center;
                }

                /* Default (unselected) tab */
                .stTabs [data-baseweb="tab"] {
                    background-color: #cccccc;  /* Light grey for unselected tabs */
                    color: #333333;             /* Dark text for readability */
                    padding: 8px 18px;
                    border-radius: 10px;
                    font-weight: 600;
                    transition: all 0.25s ease-in-out;
                    border: none;
                }

                /* Hover effect */
                .stTabs [data-baseweb="tab"]:hover {
                    background-color: #bfbfbf;
                    color: #000000;
                }

                /* Selected (active) tab */
                .stTabs [aria-selected="true"] {
                    background-color: #F4A6B8 !important;   /* Soft violet-blue */
                    color: white !important;
                    border-bottom: 3px solid #B19CD9 !important; /* Gentle pastel accent */
                    box-shadow: 0px 2px 6px rgba(108,99,255,0.3);
                }

                /* Uniform tab panel style */
                div[data-baseweb="tab-panel"] {
                    min-height: 370px !important;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    background-color: #F8F9FA;
                    border-radius: 10px;
                    padding: 15px;
                }
                </style>
                """, unsafe_allow_html=True)

                tabs = st.tabs(["Youtube", "Reddit", "Google_play_movies"])

                with tabs[0]:
                    def display_card(title, content):
                        st.markdown(f"""
                        <div style="
                            border:1px solid #ddd; border-radius:8px; padding:15px; margin-bottom:15px;
                            box-shadow:2px 2px 5px rgba(0,0,0,0.1); background-color:#fafafa;">
                            <div style="background-color: #cccccc; padding: 6px; border-radius: 8px; text-align: center;">
                                <h9 style="color: black; margin: 0;">{title}</h9>
                            </div>
                            <div style="height: 15px;"></div>
                            <p style="color:black;">{content}</p>
                        </div>
                        """, unsafe_allow_html=True)  
                    
                    def display_titles(title):
                        st.markdown(f"""
                            <div style="background-color: #2F2F30; padding: 6px; border-radius: 8px; text-align: center;">
                                <h7 style="color: white; margin: 0;"><b>{title}</b></h7>
                            </div>
                            <div style="height: 15px;"></div>
                            """,unsafe_allow_html=True)
                        
                    #--------Sentiment Analysis-------#

                    sentiment_data = llm_metrics.get("Sentiment Analysis", "N/A")

                    if sentiment_data == "N/A" or not sentiment_data:
                        st.info("No sentiment data available.")
                    else:
                        display_titles("🎭 Sentiment Analysis")

                        # Public Opinion
                        overall = sentiment_data.get("Public Opinion", {})
                        if overall:
                            formatted_overall = "<br>".join([f"• <strong>{key}</strong>: {value}" for key, value in overall.items()])
                            display_card("📊 Public Opinion", formatted_overall)
                        else:
                            display_card("📊 Public Opinion", "No data available.")

                        # Emotional Intensity
                        emotional = sentiment_data.get("Emotional Intensity", {})
                        if emotional:
                            formatted_emotional = "<br>".join([f"• <strong>{key}</strong>: {value}" for key, value in emotional.items()])
                            display_card("💫 Emotional Intensity", formatted_emotional)
                        else:
                            display_card("💫 Emotional Intensity", "No data available.")

                        display_titles("📚 Themes")

                    topics = llm_metrics.get("Themes", {}) or {}

                    if topics:
                        # Prepare weights
                        topic_weights = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))

                        # Reduce size for compact display
                        wc = WordCloud(
                            width=800,
                            height=400,
                            background_color="#f9f9f9",
                            colormap="viridis_r",
                            prefer_horizontal=0.9,
                            max_words=50,
                            max_font_size=80,
                            min_font_size=10,
                            normalize_plurals=True,
                            random_state=42,
                            scale=3
                        ).generate_from_frequencies(topic_weights)

                        # Display
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.imshow(wc, interpolation="antialiased")
                        ax.axis("off")
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.info("No Themes found for this movie.")



                    audience_data = llm_metrics.get("Audience Preferences", {})
                    if audience_data:
                        formatted_audience_data = "<br>".join([f"• <strong>{key}</strong>: {value}" for key, value in audience_data.items()])
                        display_card("💬 Audience Preferences", formatted_audience_data)
                    else:
                        display_card("💬 Audience Preferences", "No data available.")
                    
                    
                    display_card("⚖️ Expectations vs. Reality", llm_metrics.get("Expectations vs. Reality","N/A"))
                    
                    display_card("🌟 Memorable Quotes", llm_metrics.get("Memorable Quotes","N/A"))
                    
                    criticism = llm_metrics.get("Criticism", [])

                    viewer_engagement = llm_metrics.get("Viewer engagement", [])

                    cultural_insights = llm_metrics.get("Cultural Insights", {})
 
                    production_review = llm_metrics.get("Production Review", [])

                    narrative_structure = llm_metrics.get("Narrative Structure & Plot Complexity", [])

                    aesthetics = llm_metrics.get("Aesthetics", [])



                    # Utility function to display lists as HTML bullets
                    def display_bullets(title, points):

                        if points:
                            formatted = "<br>".join([f"• {p}" for p in points])
                            display_card(title, formatted)
                        else:
                            display_card(title, "No data available.")

                    # Criticism
                    display_bullets("🛑 Criticism", criticism)

                    # Viewer Engagement
                    display_bullets("👥 Viewer Engagement", viewer_engagement)


                    cultural_insights = llm_metrics.get("Cultural Insights", {})

                    if cultural_insights:
                        formatted_list = []

                        for key, value in cultural_insights.items():
                            if value:  
                                if isinstance(value, list):
                                    formatted_list.append(f"<strong>{key}</strong>")
                                    formatted_list.extend([f"• {p}" for p in value if p])
                                else:
                                    formatted_list.append(f"• <strong>{key}</strong>: {value}")

                        if formatted_list:
                            formatted = "<br>".join(formatted_list)
                            display_card("🌏 Cultural Insights", formatted)
                        else:
                            display_card("🌏 Cultural Insights", "No data available.")
                    else:
                        display_card("🌏 Cultural Insights", "No data available.")


                    # Production Review
                    display_bullets("🎬 Production Review", production_review)

                    # Narrative Structure & Plot Complexity
                    display_bullets("📖 Narrative Structure & Plot Complexity", narrative_structure)

                    # Aesthetics
                    display_bullets("🎨 Aesthetics", aesthetics)

                with tabs[1]:
                    def display_card(title, content):
                        st.markdown(f"""
                        <div style="
                            border:1px solid #ddd; border-radius:8px; padding:15px; margin-bottom:15px;
                            box-shadow:2px 2px 5px rgba(0,0,0,0.1); background-color:#fafafa;">
                            <div style="background-color: #cccccc; padding: 6px; border-radius: 8px; text-align: center;">
                                <h9 style="color: black; margin: 0;">{title}</h9>
                            </div>
                            <div style="height: 15px;"></div>
                            <p style="color:black;">{content}</p>
                        </div>
                        """, unsafe_allow_html=True)  
                    
                    def display_titles(title):
                        st.markdown(f"""
                            <div style="background-color: #2F2F30; padding: 6px; border-radius: 8px; text-align: center;">
                                <h7 style="color: white; margin: 0;"><b>{title}</b></h7>
                            </div>
                            <div style="height: 15px;"></div>
                            """,unsafe_allow_html=True)
                        
                    #--------Sentiment Analysis-------#

                    sentiment_data = llm_metrics_red.get("Sentiment Analysis", "N/A")

                    if sentiment_data == "N/A" or not sentiment_data:
                        st.info("No sentiment data available.")
                    else:
                        display_titles("🎭 Sentiment Analysis")

                        # Public Opinion
                        overall = sentiment_data.get("Public Opinion", {})
                        if overall:
                            formatted_overall = "<br>".join([f"• <strong>{key}</strong>: {value}" for key, value in overall.items()])
                            display_card("📊 Public Opinion", formatted_overall)
                        else:
                            display_card("📊 Public Opinion", "No data available.")

                        # Emotional Intensity
                        emotional = sentiment_data.get("Emotional Intensity", {})
                        if emotional:
                            formatted_emotional = "<br>".join([f"• <strong>{key}</strong>: {value}" for key, value in emotional.items()])
                            display_card("💫 Emotional Intensity", formatted_emotional)
                        else:
                            display_card("💫 Emotional Intensity", "No data available.")

                        display_titles("📚 Themes")

                    topics = llm_metrics_red.get("Themes", {}) or {}

                    if topics:
                        # Prepare weights
                        topic_weights = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))

                        # Reduce size for compact display
                        wc = WordCloud(
                            width=800,
                            height=400,
                            background_color="#f9f9f9",
                            colormap="viridis_r",
                            prefer_horizontal=0.9,
                            max_words=50,
                            max_font_size=80,
                            min_font_size=10,
                            normalize_plurals=True,
                            random_state=42,
                            scale=3
                        ).generate_from_frequencies(topic_weights)

                        # Display
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.imshow(wc, interpolation="antialiased")
                        ax.axis("off")
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.info("No Themes found for this movie.")




                    audience_data = llm_metrics_red.get("Audience Preferences", {})
                    if audience_data:
                        formatted_audience_data = "<br>".join([f"• <strong>{key}</strong>: {value}" for key, value in audience_data.items()])
                        display_card("💬 Audience Preferences", formatted_audience_data)
                    else:
                        display_card("💬 Audience Preferences", "No data available.")
                    
                    
                    display_card("⚖️ Expectations vs. Reality", llm_metrics_red.get("Expectations vs. Reality","N/A"))
                    
                    display_card("🌟 Memorable Quotes", llm_metrics_red.get("Memorable Quotes","N/A"))
                    
                    criticism = llm_metrics_red.get("Criticism", [])

                    viewer_engagement = llm_metrics_red.get("Viewer engagement", [])

                    cultural_insights = llm_metrics_red.get("Cultural Insights", {})
 
                    production_review = llm_metrics_red.get("Production Review", [])

                    narrative_structure = llm_metrics_red.get("Narrative Structure & Plot Complexity", [])

                    aesthetics = llm_metrics_red.get("Aesthetics", [])



                    # Utility function to display lists as HTML bullets
                    def display_bullets(title, points):

                        if points:
                            formatted = "<br>".join([f"• {p}" for p in points])
                            display_card(title, formatted)
                        else:
                            display_card(title, "No data available.")

                    # Criticism
                    display_bullets("🛑 Criticism", criticism)

                    # Viewer Engagement
                    display_bullets("👥 Viewer Engagement", viewer_engagement)


                    cultural_insights = llm_metrics_red.get("Cultural Insights", {})

                    if cultural_insights:
                        formatted_list = []

                        for key, value in cultural_insights.items():
                            if value:  
                                if isinstance(value, list):
                                    formatted_list.append(f"<strong>{key}</strong>")
                                    formatted_list.extend([f"• {p}" for p in value if p])
                                else:
                                    formatted_list.append(f"• <strong>{key}</strong>: {value}")

                        if formatted_list:
                            formatted = "<br>".join(formatted_list)
                            display_card("🌏 Cultural Insights", formatted)
                        else:
                            display_card("🌏 Cultural Insights", "No data available.")
                    else:
                        display_card("🌏 Cultural Insights", "No data available.")


                    # Production Review
                    display_bullets("🎬 Production Review", production_review)

                    # Narrative Structure & Plot Complexity
                    display_bullets("📖 Narrative Structure & Plot Complexity", narrative_structure)

                    # Aesthetics
                    display_bullets("🎨 Aesthetics", aesthetics)

                with tabs[2]:
                    def display_card(title, content):
                        st.markdown(f"""
                        <div style="
                            border:1px solid #ddd; border-radius:8px; padding:15px; margin-bottom:15px;
                            box-shadow:2px 2px 5px rgba(0,0,0,0.1); background-color:#fafafa;">
                            <div style="background-color: #cccccc; padding: 6px; border-radius: 8px; text-align: center;">
                                <h9 style="color: black; margin: 0;">{title}</h9>
                            </div>
                            <div style="height: 15px;"></div>
                            <p style="color:black;">{content}</p>
                        </div>
                        """, unsafe_allow_html=True)  
                    
                    def display_titles(title):
                        st.markdown(f"""
                            <div style="background-color: #2F2F30; padding: 6px; border-radius: 8px; text-align: center;">
                                <h7 style="color: white; margin: 0;"><b>{title}</b></h7>
                            </div>
                            <div style="height: 15px;"></div>
                            """,unsafe_allow_html=True)
                        
                    #--------Sentiment Analysis-------#

                    sentiment_data = llm_metrics_google.get("Sentiment Analysis", "N/A")

                    if sentiment_data == "N/A" or not sentiment_data:
                        st.info("No sentiment data available.")
                    else:
                        display_titles("🎭 Sentiment Analysis")

                        # Public Opinion
                        overall = sentiment_data.get("Public Opinion", {})
                        if overall:
                            formatted_overall = "<br>".join([f"• <strong>{key}</strong>: {value}" for key, value in overall.items()])
                            display_card("📊 Public Opinion", formatted_overall)
                        else:
                            display_card("📊 Public Opinion", "No data available.")

                        # Emotional Intensity
                        emotional = sentiment_data.get("Emotional Intensity", {})
                        if emotional:
                            formatted_emotional = "<br>".join([f"• <strong>{key}</strong>: {value}" for key, value in emotional.items()])
                            display_card("💫 Emotional Intensity", formatted_emotional)
                        else:
                            display_card("💫 Emotional Intensity", "No data available.")

                        display_titles("📚 Themes")

                    topics = llm_metrics_google.get("Themes", {}) or {}

                    if topics:
                        # Prepare weights
                        topic_weights = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))

                        # Reduce size for compact display
                        wc = WordCloud(
                            width=800,
                            height=400,
                            background_color="#f9f9f9",
                            colormap="viridis_r",
                            prefer_horizontal=0.9,
                            max_words=50,
                            max_font_size=80,
                            min_font_size=10,
                            normalize_plurals=True,
                            random_state=42,
                            scale=3
                        ).generate_from_frequencies(topic_weights)

                        # Display
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.imshow(wc, interpolation="antialiased")
                        ax.axis("off")
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.info("No Themes found for this movie.")



                    audience_data = llm_metrics_google.get("Audience Preferences", {})
                    if audience_data:
                        formatted_audience_data = "<br>".join([f"• <strong>{key}</strong>: {value}" for key, value in audience_data.items()])
                        display_card("💬 Audience Preferences", formatted_audience_data)
                    else:
                        display_card("💬 Audience Preferences", "No data available.")
                    
                    
                    display_card("⚖️ Expectations vs. Reality", llm_metrics_google.get("Expectations vs. Reality","N/A"))
                    
                    display_card("🌟 Memorable Quotes", llm_metrics_google.get("Memorable Quotes","N/A"))
                    
                    criticism = llm_metrics_google.get("Criticism", [])

                    viewer_engagement = llm_metrics_google.get("Viewer engagement", [])

                    cultural_insights = llm_metrics_google.get("Cultural Insights", {})
 
                    production_review = llm_metrics_google.get("Production Review", [])

                    narrative_structure = llm_metrics_google.get("Narrative Structure & Plot Complexity", [])

                    aesthetics = llm_metrics_google.get("Aesthetics", [])



                    # Utility function to display lists as HTML bullets
                    def display_bullets(title, points):

                        if points:
                            formatted = "<br>".join([f"• {p}" for p in points])
                            display_card(title, formatted)
                        else:
                            display_card(title, "No data available.")

                    # Criticism
                    display_bullets("🛑 Criticism", criticism)

                    # Viewer Engagement
                    display_bullets("👥 Viewer Engagement", viewer_engagement)


                    cultural_insights = llm_metrics_google.get("Cultural Insights", {})

                    if cultural_insights:
                        formatted_list = []

                        for key, value in cultural_insights.items():
                            if value:  
                                if isinstance(value, list):
                                    formatted_list.append(f"<strong>{key}</strong>")
                                    formatted_list.extend([f"• {p}" for p in value if p])
                                else:
                                    formatted_list.append(f"• <strong>{key}</strong>: {value}")

                        if formatted_list:
                            formatted = "<br>".join(formatted_list)
                            display_card("🌏 Cultural Insights", formatted)
                        else:
                            display_card("🌏 Cultural Insights", "No data available.")
                    else:
                        display_card("🌏 Cultural Insights", "No data available.")


                    # Production Review
                    display_bullets("🎬 Production Review", production_review)

                    # Narrative Structure & Plot Complexity
                    display_bullets("📖 Narrative Structure & Plot Complexity", narrative_structure)

                    # Aesthetics
                    display_bullets("🎨 Aesthetics", aesthetics)
