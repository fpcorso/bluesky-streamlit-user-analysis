import streamlit as st
import pandas as pd
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from atproto import Client
from atproto_client.exceptions import BadRequestError, InvokeTimeoutError
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt

client = Client()
client.login(st.secrets["bluesky_login"], st.secrets["bluesky_password"])

st.set_page_config(
    page_title="Bluesky User Analysis",
    page_icon="",
    layout="wide",
)

def get_did_from_handle(handle: str) -> str:
    return client.resolve_handle(handle).did

@st.cache_resource
def get_posts_from_handle(handle: str) -> list:
    """Get posts from ATProto API for user by handle."""
    did = get_did_from_handle(handle)
    max_pages = 40
    total_pages = 0
    page = client.get_author_feed(actor=did, limit=50)
    posts = page.feed

    # Keep getting posts until there are no more or until we've reached the max we want.
    while page.cursor is not None and total_pages < max_pages:
        try:
            page = client.get_author_feed(actor=did, limit=50, cursor=page.cursor)
            posts = posts + page.feed
        except InvokeTimeoutError:
            # Sometimes the Bluesky API timesout, so just try again in the next loop cycle
            pass
        total_pages += 1
    return posts

@st.cache_data
def get_posts_df_from_handle(handle: str) -> pd.DataFrame:
    """Gets the posts from ATProto API for user by handle and flattens them into a DataFrame."""
    posts = []
    raw_posts = get_posts_from_handle(handle)
    for post in raw_posts:
        main_image = None
        if post.post.embed and hasattr(post.post.embed, 'images'):
            main_image = post.post.embed.images[0].fullsize
        posts.append({
            'uri': post.post.uri,
            'author': post.post.author.handle,
            'posted_date': datetime.strptime(post.post.indexed_at, '%Y-%m-%dT%H:%M:%S.%fZ'),
            'reposted_date': datetime.strptime(post.reason.indexed_at, '%Y-%m-%dT%H:%M:%S.%fZ') if post.reason else None,
            'text': post.post.record.text,
            'likes': post.post.like_count,
            'replies': post.post.reply_count,
            'is_repost': True if post.reason else False,
            'is_reply': True if post.reply else False,
            'engagements': post.post.like_count + post.post.reply_count,
            'main_image': main_image
        })
    return pd.DataFrame.from_records(posts)


def get_total_posts_per_day(posts_df: pd.DataFrame, return_total_reposts: bool = False) -> dict:
    """Get counts of posts per day."""
    totals = {}
    for i, row in posts_df.iterrows():
        if return_total_reposts:
            date_string = row['reposted_date'].date()
        else:
            date_string = row['posted_date'].date()
        if date_string not in totals.keys():
            totals[date_string] = 1
        else:
            totals[date_string] += 1
    return totals


def get_total_posts_per_day_df(posts_df: pd.DataFrame, return_total_reposts: bool = False):
    """Create DataFrame of all dates with total number of posts posted on the date, not including reposts."""
    # Return empty DataFrame if we have 0 posts.
    if len(posts_df) == 0:
        return pd.DataFrame()

    if return_total_reposts:
        user_posts_df = posts_df[(posts_df['is_repost']) & ~(posts_df['reposted_date'].isna())].copy()
    else:
        user_posts_df = posts_df[~(posts_df['is_repost'])].copy()
    
    # If we have 0 entries after filtering, return empty DataFrame.
    if len(user_posts_df) == 0:
        return pd.DataFrame()

    # Create new DataFrame with dates and totals.
    data_df = pd.DataFrame.from_dict(get_total_posts_per_day(user_posts_df, return_total_reposts), orient='index')

    # Fill in dates that we haven't posted on
    print(data_df.index.min(), data_df.index.max())
    full_range = pd.date_range(start=data_df.index.min(), end=data_df.index.max())
    return data_df.reindex(full_range, fill_value=0).reset_index().rename(columns={'index': 'date', 0:'posts_per_day'})

def get_tags_from_feed_posts(posts) -> list:
    """Get all tags attached to all posts."""
    tags = []
    for post in posts:
        if not post.reason:
            try:
                for facet in post.post.record.facets:
                    for feature in facet.features:
                        try:
                            tags.append(feature.tag.lower())
                        except AttributeError:
                            continue
            except TypeError:
                continue
    return tags

def get_most_used_tags(handle: str):
    """Get the most used tags by a user."""
    tags = get_tags_from_feed_posts(get_posts_from_handle(handle))
    return pd.DataFrame.from_records(FreqDist(tags).most_common(10)).rename(columns={0: 'tag', 1: 'count'})

def clean_text(text):
    """Cleans text before analysis."""
    cleaned = re.sub(r'[' + string.punctuation + '’—”' + ']', "", text.lower())
    return re.sub(r'\W+', ' ', cleaned)

def create_word_frequencies(handle: str, tokens: list) -> FreqDist:
    """Convert list of words into frequency counter, removing stop words and user's handle."""
    # Add some additional stop words based on posts tested on during analysis.
    custom_stop_words = {'na', 'u', 'id', 'theres', 'ill', 'one', 'hes', 'thats', 'youre', 'could', 'go', 'going', 'want', 'didnt', 'also', 'wherever', 'aint', 'see', 'really', 'till', 'much', 'would', 'make', 'follow', 'use', 'using', 'rt', 'like', 'need', 'many', 'get', 'create', 'im', 'us', 'cant', 'got', 'read', 'dont', 'check', 'ive', 'theyre'}

    fdist = FreqDist(tokens)
    for word in (set(stopwords.words('english')).union(custom_stop_words)):
        if word in fdist:
            del fdist[word]
    
    # If the cleaned version of our handle is in our word list, remove it.
    # This occurs if the most reposted posts are ones mentioning the user.
    cleaned_handle = clean_text(handle)
    if cleaned_handle in fdist:
        del fdist[cleaned_handle]
    return fdist

def get_topics_from_string(handle: str, content: str) -> FreqDist:
    """Analyzes a text string for topics."""
    cleaned_text = clean_text(content)
    tokens = word_tokenize(cleaned_text)
    return create_word_frequencies(handle, tokens)

def get_topics_from_posts(handle: str, post_contents: pd.Series) -> FreqDist:
    posts = post_contents.to_list()
    content = ''
    for post in posts:
        content += post
    return get_topics_from_string(handle, content)

def get_wordcloud_for_posts(handle: str, post_contents: pd.Series) -> WordCloud:
    wc = WordCloud(background_color="white", max_words=30)
    wc.generate_from_frequencies(get_topics_from_posts(handle, post_contents))
    return wc

def get_most_interacted_from_posts(posts, top: int = 5):
    interactions = []
    most_interacted = []
    for post in posts:
        # Get author we reposted
        if post.reason and post.post.author.did != post.reason.by.did:
            interactions.append(post.post.author.did)
            continue

        # Get who we replied to
        try:
            if post.reply and post.reply.parent.author.did != post.post.author.did:
                interactions.append(post.reply.parent.author.did)
        except AttributeError:
            # If we are replying to post that no longer exists, getting author fails.
            pass

    most_interacted_dids = [a[0] for a in FreqDist(interactions).most_common(top)]
    for most_interacted_did in most_interacted_dids:
        user = client.get_profile(actor=most_interacted_did)
        most_interacted.append({
            'avatar': user.avatar,
            'handle': user.handle
        })
    return most_interacted

def get_most_interacted(handle: str) -> list:
    return get_most_interacted_from_posts(get_posts_from_handle(handle), 10)


st.title("Bluesky User Analysis")
st.markdown("Hey there! This Streamlit app analyzes your Bluesky posts (up to your 2,000 most recent posts). I created it while exploring the ATProto (what Bluesky runs on) API and SDKs and thought I'd publish it in case any one else found it interesting.")
st.markdown("To begin, enter your Bluesky handle. :point_down:")

user_input = st.text_input(
    label="Input your Bluesky (ATProto) handle (e.g. fpcorso.bsky.social)",
    value=""
)
if not user_input:
    st.warning("You must enter a handle in order to show any analysis.")
else:
    try:
        with st.spinner(text="Loading your posts..."):
            data_df = get_posts_df_from_handle(user_input)
    except (BadRequestError, InvokeTimeoutError) as e:
        print(e)
        try:
            st.error(f"Loading failed. This app encountered an error when loading posts through Bluesky API: {e.response.content.message}. Please try again.")
        except: 
            st.error("Loading failed. This app encountered an error when loading posts through Bluesky API. Please try again.")
        st.stop()
    
    if len(data_df) == 0:
        st.info("This user does not have any posts to analyze.")
        st.stop()
    
    try:
        with st.spinner(text="Analyzing your posts..."):
            posts_per_day_df = get_total_posts_per_day_df(data_df)
            reposts_per_day_df = get_total_posts_per_day_df(data_df, True)
            tags = get_most_used_tags(user_input)
            most_interacted_with = get_most_interacted(user_input)
    except (BadRequestError, InvokeTimeoutError, ValueError) as e:
        print(e)
        st.error("Analyzing failed. Please try again.")
        st.stop()

    st.divider()
    st.header("A look at you")

    with st.container(horizontal=True, horizontal_alignment="distribute"):
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric(
                "Total Posts",
                len(data_df[~(data_df['is_repost']) & ~(data_df['is_reply'])]),
            )

        with metric_cols[1]:
            st.metric(
                "Total Replies",
                len(data_df[~(data_df['is_repost']) & (data_df['is_reply'])]),
            )

        with metric_cols[2]:
            st.metric(
                "Total Reposts",
                len(data_df[(data_df['is_repost'])]),
            )

    st.subheader('Posts per day')
    st.markdown('Includes posts and replies')
    if len(posts_per_day_df) == 0:
        st.info("You have not posted anything yet.")
    else:
        st.line_chart(posts_per_day_df, y='posts_per_day', x='date', x_label='Date', y_label='Posts per day')

    column1, column2 = st.columns(2)
    with column1:
        st.subheader("Tags you use the most")
        st.markdown("These are your most used hashtags")
        if len(tags) == 0:
            st.info("You have not used any tags")
        else:
            st.bar_chart(tags, y='count', x='tag', sort='-count', horizontal=True)

    with column2:
        st.subheader("Words you use the most")
        st.markdown("These are the words you use the most in your posts and replies")
        if len(posts_per_day_df) == 0:
            st.info("You have not posted anything yet.")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(get_wordcloud_for_posts(user_input, data_df[~(data_df['is_repost'])]['text']), interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
    
    st.subheader('Your top posts')
    st.markdown('Your posts that received the most likes and replies')
    with st.container(horizontal=True):
        for i, row in data_df[~(data_df['is_repost'])].sort_values(by='engagements', ascending=False)[:3].iterrows():
            with st.container(border=True):
                st.markdown(row['text'])
                if row['main_image']:
                    st.image(row['main_image'])
                st.caption(f"Posted on {row['posted_date'].date()}")
                uri_parts = row['uri'][5:].split('/')
                if len(uri_parts) == 3:
                    st.markdown(f"[View on Bluesky](https://bsky.app/profile/{uri_parts[0]}/post/{uri_parts[2]})")

    st.divider()
    st.header("A look at those around you")
    st.subheader("People you interact with the most")
    st.markdown("These are the people you repost or reply to the most")
    interactions_column_1, interactions_column_2 = st.columns(2)
    with interactions_column_1:
        for interacted_with in most_interacted_with[:5]:
            interaction_column_space, interaction_column_1, interaction_column_2 = st.columns((0.1, 0.5, 2))
            with interaction_column_1:
                st.image(interacted_with['avatar'])
            with interaction_column_2:
                st.markdown(interacted_with['handle'])
    with interactions_column_2:
        for interacted_with in most_interacted_with[5:]:
            interaction_column_space, interaction_column_1, interaction_column_2 = st.columns((0.1, 0.5, 2))
            with interaction_column_1:
                st.image(interacted_with['avatar'])
            with interaction_column_2:
                st.markdown(interacted_with['handle'])
    
    interactions_analysis_column_1, interactions_analysis_column_2 = st.columns(2)
    with interactions_analysis_column_1:
        st.subheader('Reposts per day')
        if len(reposts_per_day_df) == 0:
            st.info("You have not reposted anything yet.")
        else:
            st.line_chart(reposts_per_day_df, y='posts_per_day', x='date', x_label='Date', y_label='Reposts per day')
    with interactions_analysis_column_2:
        st.subheader("Topics of reposts")
        st.markdown("These are the words used the most in posts that you repost")
        if len(reposts_per_day_df) == 0:
            st.info("You have not reposted anything yet.")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(get_wordcloud_for_posts(user_input, data_df[(data_df['is_repost'])]['text']), interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

st.divider()
st.markdown("Thanks for checking out my Streamlit app! If you have any feedback or questions, feel free to reach out [on my website](https://frankcorso.me) or [on Bluesky](https://bsky.app/profile/fpcorso.bsky.social).")