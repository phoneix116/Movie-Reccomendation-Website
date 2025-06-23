from flask import Flask, request, jsonify, render_template, send_from_directory, redirect
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import os
import json
import logging
import datetime
import random
import time
import sys
from flask_cors import CORS  # Import CORS support
from functools import wraps  # For decorator functions

# Check for fast startup mode
FAST_STARTUP = "--fast" in sys.argv

# Define log filename
log_filename = f"activity_log_{datetime.datetime.now().strftime('%Y%m%d')}.log"

app = Flask(__name__, 
             static_folder="static",  # Folder for static assets (CSS, JS, etc.)
             template_folder="templates")  # Folder for HTML templates

# Enable CORS for all routes with more permissive settings
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": "*"}})

# Add JSONP support with a decorator
def jsonpify(func):
    """Decorator to support JSONP requests by wrapping the response in a callback if specified"""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        callback = request.args.get('callback', False)
        if callback:
            # Get the JSON response
            resp = func(*args, **kwargs)
            # Convert to a string
            resp_data = resp.get_data(as_text=True)
            # Wrap in the callback
            content = f"{callback}({resp_data})"
            # Create a new response with proper JSONP headers
            jsonp_resp = app.response_class(
                response=content,
                status=resp.status_code,
                mimetype='application/javascript'
            )
            return jsonp_resp
        else:
            # Regular JSON response
            return func(*args, **kwargs)
    return decorated_function

# Clear the log file to start fresh
with open(log_filename, "w") as f:
    f.write(f"=== Movie Recommendation System Log {datetime.datetime.now()} ===\n")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a"  # Append mode
)    # Create a recommendation cache to improve performance
recommendation_cache = {}
genre_search_cache = {}  # Add cache for genre searches

# Add a handler to also print logs to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger("").addHandler(console_handler)

# Log application startup
logging.info("Application starting")

# Define the port to use
SERVER_PORT = 3000

# Flag to enable/disable expensive operations
ENABLE_ANNOY = False  # Changed to False by default - we'll enable it manually if needed
ENABLE_CONTENT_BASED = True  # Set to False to completely disable content-based recommendations

# Define the genre list
all_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

try:
    # Check for cached preprocessed data first
    df_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "df_cache.pkl")
    df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "df.csv")
    
    # Variable to track if we loaded from cache successfully
    loaded_from_cache = False
    
    # Try to load the cached dataset if not in FAST_STARTUP mode
    if not FAST_STARTUP and os.path.exists(df_cache_path):
        try:
            logging.info(f"Attempting to load cached dataframe from {df_cache_path}")
            start_time = time.time()
            with open(df_cache_path, 'rb') as f:
                df_data = pickle.load(f)
                
            if 'df' in df_data and 'X_scaled' in df_data and 'feature_cols' in df_data:
                df = df_data['df']
                X_scaled = df_data['X_scaled']
                feature_cols = df_data['feature_cols']
                
                load_time = time.time() - start_time
                logging.info(f"Successfully loaded cached dataframe with {len(df)} rows in {load_time:.2f} seconds")
                
                # Skip to after the dataframe preprocessing
                logging.info("Using pre-processed cached data, skipping preprocessing steps")
                
                # We need the cluster distribution for later
                if 'cluster' in df.columns:
                    cluster_counts = df['cluster'].value_counts()
                    logging.info(f"Cluster distribution: {cluster_counts.to_dict()}")
                  # Mark that we've loaded from cache successfully
                loaded_from_cache = True
                
            else:
                logging.info("Cache file exists but doesn't contain required data, loading from CSV...")
        except Exception as cache_error:
            if not isinstance(cache_error, StopIteration):  # We no longer use StopIteration
                logging.error(f"Error loading cached dataframe: {str(cache_error)}")
                logging.info("Falling back to CSV loading")
    
    # Only load from CSV if we didn't successfully load from cache
    if not loaded_from_cache:
        logging.info(f"Loading dataframe from {df_path}")
        
        # Try to load the df dataset
        try:
            if os.path.exists(df_path):
                if FAST_STARTUP:
                    # Ultra-fast startup mode - only load a small subset of data
                    logging.info("FAST STARTUP MODE: Loading only a small subset of data")
                    df = pd.read_csv(df_path, nrows=10000)  # Only read first 10,000 rows
                    logging.info(f"Successfully loaded small dataframe with {len(df)} rows (FAST MODE)")
                else:                    # Normal mode - load the full dataset
                    start_time = time.time()
                    df = pd.read_csv(df_path)
                    load_time = time.time() - start_time
                    logging.info(f"Successfully loaded full dataframe with {len(df)} rows from df.csv in {load_time:.2f} seconds")
            else:
                raise FileNotFoundError(f"df.csv not found at {df_path}")
        except Exception as df_error:
            logging.error(f"Error loading df.csv: {str(df_error)}")
            
            # Create a minimal dataset for testing
            logging.info("Creating minimal movie dataset for testing")
            df = pd.DataFrame({
            'movieId': range(1, 11),
            'title': [
                'Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)',
                'Waiting to Exhale (1995)', 'Father of the Bride Part II (1995)',
                'Heat (1995)', 'Sabrina (1995)', 'Tom and Huck (1995)',
                'Sudden Death (1995)', 'GoldenEye (1995)'
            ],
            'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy',
                      'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy',
                      'Action|Crime|Thriller', 'Comedy|Romance', 'Adventure|Children',
                      'Action', 'Action|Adventure|Thriller'],
            'year': [1995, 1995, 1995, 1995, 1995, 1995, 1995, 1995, 1995, 1995],
            'cluster': [0, 1, 2, 2, 2, 3, 2, 1, 3, 3],
            'rating': [4.5, 3.8, 3.0, 3.5, 3.2, 4.2, 3.1, 2.9, 3.0, 4.0]
        })
    
    # Extract year from title if not present
    if 'year' not in df.columns:
        logging.info("Extracting year from movie titles")
        # Use regex to extract year from titles like "Movie Title (1997)"
        df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype('float')
        
    # Create cluster column if not present (using KMeans)
    if 'cluster' not in df.columns:
        try:
            logging.info("Generating clusters for movies")
            from sklearn.cluster import KMeans
            
            # Generate features from genres
            for genre in all_genres:
                df[genre] = df['genres'].apply(lambda x: 1 if genre in str(x) else 0)
            
            # Use genre features and year for clustering
            cluster_features = all_genres + ['year'] if 'year' in df.columns else all_genres
            
            # Fill NAs for clustering
            X = df[cluster_features].fillna(0)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=10, random_state=42)
            df['cluster'] = kmeans.fit_predict(X)
            logging.info("Successfully generated clusters")
        except Exception as cluster_error:
            logging.error(f"Error generating clusters: {str(cluster_error)}")
            # Random clusters as fallback
            df['cluster'] = np.random.randint(0, 10, size=len(df))
            logging.warning("Using random clusters as fallback")
    
    # Log the cluster distribution
    if 'cluster' in df.columns:
        cluster_counts = df['cluster'].value_counts()
        logging.info(f"Cluster distribution: {cluster_counts.to_dict()}")
        
    # Fill NA values
    df.fillna(0, inplace=True)
    
    # Prepare X_scaled for recommendation function
    try:
        # Extract genre features if they don't exist
        for genre in all_genres:
            if genre not in df.columns:
                df[genre] = df['genres'].apply(lambda x: 1 if genre in str(x) else 0)
        
        feature_cols = all_genres + ['year'] if 'year' in df.columns else all_genres
        X = df[feature_cols].copy()
        X.fillna(0, inplace=True)
          # Apply scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logging.info("Feature matrix prepared for recommendations")
        
        # Save the processed dataframe to cache if not in FAST_STARTUP mode and we didn't load from cache
        if not FAST_STARTUP and not loaded_from_cache:
            try:
                df_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "df_cache.pkl")
                logging.info(f"Saving processed dataframe to cache: {df_cache_path}")
                with open(df_cache_path, 'wb') as f:
                    pickle.dump({
                        'df': df,
                        'X_scaled': X_scaled,
                        'feature_cols': feature_cols
                    }, f)
                logging.info("Dataframe cache saved successfully")
            except Exception as cache_error:
                logging.error(f"Failed to save dataframe cache: {str(cache_error)}")
    except Exception as feature_error:
        logging.error(f"Error preparing feature matrix: {str(feature_error)}")
        X_scaled = None

except Exception as init_error:
    logging.critical(f"Critical error during initialization: {str(init_error)}")
    # Create minimal dataset to avoid crashes
    df = pd.DataFrame({
        'movieId': range(1, 6),
        'title': ['Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)',
                 'Waiting to Exhale (1995)', 'Father of the Bride Part II (1995)'],
        'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy',
                  'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy'],
        'rating': [4.5, 3.8, 3.0, 3.5, 3.2],
        'year': [1995, 1995, 1995, 1995, 1995],
        'cluster': [0, 1, 2, 2, 2]
    })
    logging.info("Created emergency fallback dataset")
    X_scaled = None

# Build fast lookup indices for titles and genres
logging.info("Building fast lookup indices for titles and genres")

# Check if we have cached indices we can load
indices_pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indices_cache.pkl")
indices_loaded = False

# Only try to load cache if we're not in fast startup mode (which uses a subset of data)
if not FAST_STARTUP and os.path.exists(indices_pickle_path):
    try:
        logging.info(f"Attempting to load cached indices from {indices_pickle_path}")
        with open(indices_pickle_path, 'rb') as f:
            cached_indices = pickle.load(f)
            
            # Verify the cache is compatible with our current dataset
            if ('df_rows' in cached_indices and 
                'title_index' in cached_indices and 
                'title_without_year_index' in cached_indices and
                'title_words_index' in cached_indices and 
                'genre_index' in cached_indices and
                cached_indices['df_rows'] == len(df)):
                
                # Cache is valid, use it
                title_index = cached_indices['title_index']
                title_without_year_index = cached_indices['title_without_year_index']
                title_words_index = cached_indices['title_words_index']
                genre_index = cached_indices['genre_index']
                  # Get counts for logging
                unique_titles = set()
                for titles in title_index.values():
                    for idx in titles:
                        try:
                            # Check if the index is valid
                            if idx >= 0 and idx < len(df):
                                title = df.iloc[idx]['title']
                                # Make sure title is a string before calling lower()
                                if isinstance(title, str):
                                    unique_titles.add(title.lower())
                                else:
                                    # Convert non-string titles to string
                                    unique_titles.add(str(title).lower())
                        except Exception as title_error:
                            logging.warning(f"Error adding title at index {idx} to unique titles set: {str(title_error)}")
                
                unique_genres = set(genre_index.keys())
                
                logging.info(f"Successfully loaded cached indices for {len(unique_titles)} titles and {len(unique_genres)} genres")
                indices_loaded = True
            else:
                logging.info("Cached indices not compatible with current dataset, rebuilding...")
    except Exception as e:
        logging.error(f"Error loading cached indices: {str(e)}")
        logging.info("Will rebuild indices from scratch")

# If we couldn't load cached indices, build them
if not indices_loaded:
    title_index = {}  # Full title -> list of indices
    title_without_year_index = {}  # Title without year -> list of indices
    title_words_index = {}  # Words in title -> list of indices
    genre_index = {}  # Genre -> list of indices

    # Count unique titles for logging
    unique_titles = set()
    unique_genres = set()

    # Process in batches for large datasets to avoid memory spikes
    batch_size = 100000  # Process this many rows at once
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division

    logging.info(f"Building indices in {num_batches} batches (batch size: {batch_size})")
    
    # Process in batches
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_rows)
        
        if batch_num % 10 == 0 or batch_num == num_batches - 1:  # Log every 10 batches or the last batch
            logging.info(f"Processing batch {batch_num+1}/{num_batches} (rows {start_idx}-{end_idx})")
        
        # Get a slice of the dataframe
        df_batch = df.iloc[start_idx:end_idx]
        
        for idx, row in df_batch.iterrows():
            # Process title
            if 'title' in row and pd.notnull(row['title']):
                # Get the full lowercase title
                full_title = str(row['title']).lower()
                unique_titles.add(full_title)
                
                # Add to full title index
                if full_title not in title_index:
                    title_index[full_title] = []
                title_index[full_title].append(idx)
                
                # Extract title without year and add to that index
                title_no_year = full_title.split('(')[0].strip()
                if title_no_year not in title_without_year_index:
                    title_without_year_index[title_no_year] = []
                title_without_year_index[title_no_year].append(idx)
                
                # Add individual words to word index for more flexible searching
                for word in title_no_year.split():
                    if len(word) > 2:  # Only index words with more than 2 characters
                        if word not in title_words_index:
                            title_words_index[word] = []
                        title_words_index[word].append(idx)
            
            # Process genres
            if 'genres' in row and pd.notnull(row['genres']):
                genres = str(row['genres']).split('|')
                for genre in genres:
                    genre = genre.strip().lower()
                    unique_genres.add(genre)
                    if genre:
                        if genre not in genre_index:
                            genre_index[genre] = []
                        genre_index[genre].append(idx)

    logging.info(f"Indexed {len(unique_titles)} unique titles and {len(unique_genres)} unique genres")
    logging.info(f"Created {len(title_index)} full title entries, {len(title_without_year_index)} title-without-year entries, and {len(title_words_index)} word entries")
    logging.info(f"Created {len(genre_index)} genre entries")

    # Save indices to cache (but only if not in FAST_STARTUP mode which uses a subset)
    if not FAST_STARTUP:
        try:
            import pickle
            logging.info(f"Saving indices to cache file: {indices_pickle_path}")
            with open(indices_pickle_path, 'wb') as f:
                pickle.dump({
                    'df_rows': len(df),
                    'title_index': title_index,
                    'title_without_year_index': title_without_year_index,
                    'title_words_index': title_words_index,
                    'genre_index': genre_index
                }, f)
            logging.info("Indices cache saved successfully")
        except Exception as e:
            logging.error(f"Failed to save indices cache: {str(e)}")
else:
    logging.info("Using cached indices, skipping index building")

# Helper function to prepare data for API responses
def prepare_movie_data(df_subset):
    """Prepares a subset of movies for API response by selecting relevant columns."""
    available_columns = ['movieId', 'title', 'genres', 'year', 'rating']
    available_columns = [col for col in available_columns if col in df_subset.columns]
    
    return df_subset[available_columns]

# Setup variables for Annoy - but don't load the module unless needed
annoy_index = None
annoy_index_path = 'annoy_index.ann'
num_features = len(all_genres) + (1 if 'year' in df.columns else 0)
max_annoy_items = 100000  # Maximum items to include in Annoy index

# Function to build a partial Annoy index when needed
def build_partial_annoy_index():
    global annoy_index
    
    if not ENABLE_ANNOY:
        logging.info("Annoy is disabled - skipping index building")
        return None
        
    if annoy_index is not None:
        return annoy_index  # Already built
    
    try:
        # First try to import Annoy here - this way we don't load it at startup
        from annoy import AnnoyIndex
        logging.info("Building partial Annoy index on demand (first-time request)")
        
        # Set a timeout for building
        start_time = time.time()
        max_build_time = 10  # seconds - reduced from 20 to 10 for faster response
        
        # Create the index
        temp_index = AnnoyIndex(num_features, 'euclidean')
        
        # Sample size reduced for faster builds
        sample_size = min(10000, len(df))  # Use at most 10,000 items
        
        # Simple random sampling (much faster than stratified sampling)
        if len(df) <= sample_size:
            sample_indices = df.index.tolist()
        else:
            sample_indices = df.sample(sample_size).index.tolist()
        
        # Build the index with our sampled items
        added_count = 0
        for idx in sample_indices:
            if time.time() - start_time > max_build_time:
                logging.warning(f"Annoy index building timed out after {max_build_time} seconds")
                break
                
            row = df.loc[idx]
            features = [row[genre] if genre in row else 0 for genre in all_genres]
            if 'year' in df.columns:
                features.append(row['year'])
            temp_index.add_item(idx, features)
            added_count += 1
        
        # Only build if we added a reasonable number of items
        if added_count > 100:  # Reduced threshold from 1000 to 100
            temp_index.build(5)  # Reduced from 10 trees to 5 for speed
            logging.info(f'Partial Annoy index built with {added_count} items')
            return temp_index
        else:
            logging.warning(f'Not enough items added to Annoy index ({added_count}), using fallback method')
            return None
            
    except Exception as e:
        logging.error(f'Error building Annoy index: {e}')
        return None

def recommend_movies(title, df, X_scaled=None, n_recommendations=10):
    """
    Recommend movies based on a title using either content-based filtering 
    or cluster-based recommendations
    """
    logging.info(f"Getting recommendations for: {title}")
      # Set a timeout for recommendation processing
    import time
    start_time = time.time()
    max_processing_time = 30  # Maximum processing time in seconds
    
    if title in recommendation_cache:
        logging.info(f"Using cached recommendations for: {title}")
        return recommendation_cache[title]
    try:
        # Get the title without year for better searching
        search_title = title.split('(')[0].strip().lower()
        movie_indices = []
        
        # Try matching strategies in order of precision
        
        # 1. First try exact match on full title
        if title.lower() in title_index:
            movie_indices = title_index[title.lower()]
            logging.info(f"Found exact title match for: {title}")
        
        # 2. Then try exact match on title without year
        elif search_title in title_without_year_index:
            movie_indices = title_without_year_index[search_title]
            logging.info(f"Found title-without-year match for: {search_title}")
            
        # 3. Try partial matches on full titles
        else:
            # Collect possible matches from word index
            candidate_indices = set()
            words = search_title.split()
            
            # If we have enough words, use word index for speed
            if len(words) > 1:
                for word in words:
                    if len(word) > 2 and word in title_words_index:
                        for idx in title_words_index[word]:
                            candidate_indices.add(idx)
                
                # If we found candidate movies via word index, check them
                if candidate_indices:
                    for idx in candidate_indices:
                        movie_title = df.iloc[idx]['title'].lower()
                        # Check if search_title is a substantial part of the movie title
                        if search_title in movie_title:
                            movie_indices.append(idx)
            
            # If we still didn't find matches, do a more comprehensive search
            if not movie_indices:
                logging.info(f"No quick matches found, performing comprehensive search for: {search_title}")
                for t, indices in title_index.items():
                    if search_title in t:
                        movie_indices.extend(indices)
                        if len(movie_indices) >= 10:  # Limit to avoid too many matches
                            break
        
        # If still no matches, log the failure
        if not movie_indices:
            logging.warning(f"Movie not found after all search attempts: {title}")
            return f"Movie not found: {title}"
        
        # Use the first match (most relevant)
        movie_idx = movie_indices[0]
        movie = df.iloc[movie_idx]
        logging.info(f"Found movie: {movie['title']} (index: {movie_idx})")
        original_genres = movie['genres'].split('|') if 'genres' in movie and movie['genres'] else []
          # Skip Annoy if completely disabled via flag
        if ENABLE_ANNOY:
            # Try to use Annoy for fast similarity search (lazy loading)
            try:
                # Only try to build/use Annoy for the first few requests to avoid slowing down the server
                if title not in recommendation_cache and len(recommendation_cache) < 5:  # Reduced from 20 to 5
                    # Build the index lazily only when needed
                    if annoy_index is None:
                        annoy_index = build_partial_annoy_index()
                    
                    # If we have a valid index and the movie is in the index
                    if annoy_index is not None:
                        try:
                            # Check if the movie's index is in the Annoy index without actually raising exceptions
                            if movie_idx < annoy_index.get_n_items():
                                # Get similar movies
                                n_search = n_recommendations * 2
                                neighbor_indices = annoy_index.get_nns_by_item(movie_idx, n_search + 1)[1:]
                                all_recommendations = df.iloc[neighbor_indices]
                                seen_titles = set()
                                recommendations = pd.DataFrame()
                                
                                for _, rec in all_recommendations.iterrows():
                                    rec_title_lower = rec['title'].lower() if isinstance(rec['title'], str) else ''
                                    if rec_title_lower not in seen_titles and len(recommendations) < n_recommendations:
                                        seen_titles.add(rec_title_lower)
                                        recommendations = pd.concat([recommendations, rec.to_frame().T])
                                
                                if len(recommendations) < n_recommendations:
                                    logging.info("Adding more movies to content-based recommendations")
                                    recommendations = get_additional_recommendations(recommendations, movie, original_genres, n_recommendations, df, seen_titles)
                                
                                logging.info(f"Returning {len(recommendations)} unique content-based recommendations (Annoy)")
                                recommendation_cache[title] = recommendations
                                return recommendations
                            else:
                                logging.info(f"Movie {movie_idx} not in Annoy index (index size: {annoy_index.get_n_items()})")
                        except Exception as rec_error:
                            logging.info(f"Movie index error: {str(rec_error)}")
                            # Fall through to standard recommendation methods
            except Exception as annoy_error:
                logging.error(f"Error in Annoy recommendation setup: {str(annoy_error)}")
        
        # If we have content-based features, use them
        if X_scaled is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                  # Get index of the movie
                idx = movie.name
                
                # Get feature vector for the movie
                movie_vec = X_scaled[idx].reshape(1, -1)
                
                # Check if we've exceeded the time limit
                if time.time() - start_time > max_processing_time:
                    logging.warning(f"Recommendation process timed out after {max_processing_time} seconds")
                    # Fall back to a simple genre-based approach
                    cluster_id = movie['cluster']
                    logging.info(f"Falling back to cluster-based recommendation for cluster {cluster_id}")
                    
                    # Get other movies in the same cluster
                    cluster_movies = df[df['cluster'] == cluster_id].head(n_recommendations * 2)
                    
                    # Exclude the requested movie
                    cluster_movies = cluster_movies[cluster_movies['title'] != movie['title']]
                    
                    # Sort by rating if available
                    if 'rating' in cluster_movies:
                        cluster_movies = cluster_movies.sort_values('rating', ascending=False)
                    
                    # Take just what we need
                    recommendations = cluster_movies.head(n_recommendations)
                    
                    logging.info(f"Returning {len(recommendations)} fallback cluster-based recommendations")
                    recommendation_cache[title] = recommendations
                    return recommendations
                  # Calculate cosine similarity
                logging.info("Calculating cosine similarities")
                
                # To improve performance, limit the number of movies we calculate similarity for
                # Instead of using the entire dataset, use a subset
                max_movies_for_similarity = min(100000, len(X_scaled))
                
                if len(X_scaled) > max_movies_for_similarity:
                    logging.info(f"Using subset of {max_movies_for_similarity} movies for similarity calculation")
                    # Get indices of movies from the same cluster and with similar genres
                    indices_to_use = []
                    
                    # First add movies from same cluster
                    cluster_id = movie['cluster']
                    cluster_indices = df[df['cluster'] == cluster_id].index.tolist()
                    indices_to_use.extend(cluster_indices[:min(10000, len(cluster_indices))])
                    
                    # Add movies with matching genres
                    if original_genres and len(original_genres) > 0:
                        for genre in original_genres:
                            genre_indices = df[df['genres'].astype(str).str.contains(genre, na=False)].index.tolist()
                            indices_to_use.extend(genre_indices[:min(10000, len(genre_indices))])
                    
                    # Add some random movies to ensure diversity
                    remaining_slots = max_movies_for_similarity - len(indices_to_use)
                    if remaining_slots > 0:
                        all_indices = set(range(len(X_scaled)))
                        used_indices = set(indices_to_use)
                        available_indices = list(all_indices - used_indices)
                        import random
                        random_indices = random.sample(available_indices, min(remaining_slots, len(available_indices)))
                        indices_to_use.extend(random_indices)
                    
                    # Remove duplicates
                    indices_to_use = list(set(indices_to_use))
                    
                    # Calculate similarity only for the selected subset
                    subset_X_scaled = X_scaled[indices_to_use]
                    sim_scores_subset = cosine_similarity(movie_vec, subset_X_scaled).flatten()
                    
                    # Map the scores back to original indices
                    sim_scores = np.zeros(len(X_scaled))
                    for i, idx in enumerate(indices_to_use):
                        sim_scores[idx] = sim_scores_subset[i]
                else:
                    # Calculate for all movies if the dataset isn't too large
                    sim_scores = cosine_similarity(movie_vec, X_scaled).flatten()
                
                # Get more similar movies than needed to handle duplicates
                extra_count = n_recommendations * 2  # Get 2x recommendations to allow for duplicates
                sim_indices = sim_scores.argsort()[::-1][1:extra_count+1]
                
                # Get recommended movies
                all_recommendations = df.iloc[sim_indices]
                logging.info(f"Generated {len(all_recommendations)} candidate content-based recommendations")
                
                # Filter out duplicate titles (case insensitive)
                seen_titles = set()
                recommendations = pd.DataFrame()
                
                for _, rec in all_recommendations.iterrows():
                    rec_title_lower = rec['title'].lower() if isinstance(rec['title'], str) else ''
                    if rec_title_lower not in seen_titles and len(recommendations) < n_recommendations:
                        seen_titles.add(rec_title_lower)
                        recommendations = pd.concat([recommendations, rec.to_frame().T])
                
                # If we still need more recommendations (less than requested)
                if len(recommendations) < n_recommendations:
                    logging.info("Adding more movies to content-based recommendations")
                    get_additional_recommendations(recommendations, movie, original_genres, n_recommendations, df, seen_titles)
                
                logging.info(f"Returning {len(recommendations)} unique content-based recommendations")
                return recommendations
                
            except Exception as rec_error:
                logging.error(f"Error in content-based recommendation: {str(rec_error)}")
                # Fall back to cluster recommendations
        
        # Use cluster-based recommendation if content-based fails or is not available
        cluster_id = movie['cluster']
        logging.info(f"Using cluster-based recommendation for cluster {cluster_id}")
        
        # Get other movies in the same cluster
        cluster_movies = df[df['cluster'] == cluster_id]
        
        # Exclude the requested movie
        cluster_movies = cluster_movies[cluster_movies['title'] != movie['title']]
        
        # Sort by rating if available
        if 'rating' in cluster_movies:
            cluster_movies = cluster_movies.sort_values('rating', ascending=False)
        
        # Get more movies than needed to handle duplicates
        extra_count = n_recommendations * 2
        all_recommendations = cluster_movies.head(extra_count)
        
        # Filter out duplicate titles (case insensitive)
        seen_titles = set()
        recommendations = pd.DataFrame()
        
        for _, rec in all_recommendations.iterrows():
            rec_title_lower = rec['title'].lower() if isinstance(rec['title'], str) else ''
            if rec_title_lower not in seen_titles and len(recommendations) < n_recommendations:
                seen_titles.add(rec_title_lower)
                recommendations = pd.concat([recommendations, rec.to_frame().T])
        
        # If we don't have enough recommendations, add movies with similar genres
        if len(recommendations) < n_recommendations:
            logging.info("Adding movies with similar genres to recommendations")
            get_additional_recommendations(recommendations, movie, original_genres, n_recommendations, df, seen_titles)
        
        logging.info(f"Returning {len(recommendations)} unique cluster-based recommendations")
        
        # Cache the recommendations for future use
        recommendation_cache[title] = recommendations
        
        return recommendations
        
    except Exception as e:
        logging.error(f"Error in recommendation function: {str(e)}")
        return f"Error generating recommendations: {str(e)}"

# Serve static files
@app.route('/styles.css')
def styles_css():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "styles.css")

@app.route('/search_new.js')
def search_new_js():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "search_new.js")

@app.route('/static/<path:filename>')
def serve_static(filename):
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    return send_from_directory(static_path, filename)

@app.route('/moviejs.js')
def moviejs_js():
    # First try to serve from static folder
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    if os.path.exists(os.path.join(static_path, "moviejs.js")):
        return send_from_directory(static_path, "moviejs.js")
    else:
        # Fall back to root directory
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "moviejs.js")

@app.route('/search_new_fixed.js')
def search_new_fixed_js():
    # Return an empty script to avoid errors, we're now using search-results.js instead
    return """
    // This file is deprecated, using search-results.js instead
    console.log("Using updated search-results.js file instead of search_new_fixed.js");
    """

@app.route('/static/js/search_new_fixed.js')
def static_search_new_fixed_js():
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "js"), "search_new_fixed.js")
    
@app.route('/static/js/search-results.js')
def static_search_results_js():
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "js"), "search-results.js")

@app.route("/search_working.js")
def search_working_js():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "search_working.js")

@app.route("/home-search.js")
def home_search_js():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "home-search.js")

@app.route("/search_connection_test.html")
def search_connection_test():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "search_connection_test.html")

@app.route("/api_debug.html")
def api_debug():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "api_debug.html")

@app.route("/direct_search_test.html")
def direct_search_test():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "direct_search_test.html")
    
@app.route("/integration_test.html")
def integration_test():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "integration_test.html")
    
@app.route("/movie-details.js")
def movie_details_js():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "movie-details.js")

@app.route("/movie-details-fixed.js")
def movie_details_fixed_js():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "movie-details-fixed.js")

@app.route("/movie-details-debug.js")
def movie_details_debug_js():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "movie-details-debug.js")
    
@app.route("/movie-details-v2.js")
def movie_details_v2_js():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "movie-details-v2.js")

@app.route("/movie-details-v2.html")
def movie_details_v2_html():
    # Redirect to home page
    from flask import redirect
    title = request.args.get("title", "")
    if title:
        return redirect(f"/search-results.html?q={title}&recommend=true")
    return redirect("/")
    
@app.route("/test_api.html")
def test_api_html():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "test_api.html")
    
@app.route("/movie_details_diagnostics.html")
def movie_details_diagnostics_html():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "movie_details_diagnostics.html")
    
@app.route("/server_check.html")
def server_check_html():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "server_check.html")

@app.route("/check_server.js")
def check_server_js():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "check_server.js")

@app.route("/api-status")
@jsonpify
def api_status():
    """Simple API status endpoint that returns a success response for connectivity testing"""
    # Log that the API status was checked
    logging.info("API status endpoint accessed")
    return jsonify({
        "status": "online",
        "service": "Movie Recommendation API",
        "version": "1.1.0",
        "timestamp": datetime.datetime.now().isoformat(),
        "endpoints": [
            "/search", 
            "/recommend",
            "/movie-details", 
            "/popular-movies",
            "/movies-by-genre",
            "/performance-stats",
            "/clear-cache"
        ]
    })

@app.route("/clear-cache", methods=["POST", "GET"])
def clear_cache():
    """Clear the recommendation cache"""
    global recommendation_cache
    cache_size_before = len(recommendation_cache)
    recommendation_cache = {}
    logging.info(f"Recommendation cache cleared. {cache_size_before} entries removed.")
    return jsonify({
        "status": "success",
        "message": f"Cache cleared. {cache_size_before} entries removed."
    })
    
@app.route("/")
def home():
    # Try to serve index.html from root first
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_index_path = os.path.join(root_dir, "index.html")
    
    if os.path.exists(root_index_path):
        return send_from_directory(root_dir, "index.html")
    else:
        return render_template("index.html")  # Fall back to template
        
@app.route("/index.html")
def index_html():    # Check if index.html exists in the root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_index_path = os.path.join(root_dir, "index.html")
    
    if os.path.exists(root_index_path):
        return send_from_directory(root_dir, "index.html")
    else:
        return render_template("index.html")  # Fall back to template
    
@app.route("/search", methods=["POST", "GET"])
@jsonpify
def search():
    """Endpoint to search for movies or get recommendations"""
    # Handle both POST and GET requests
    if request.method == "POST":
        try:
            data = request.get_json(silent=True) or {}
        except Exception as e:
            data = {}
            logging.error(f"Error parsing JSON data: {str(e)}")
    else:  # GET request
        data = {}  # Initialize as empty dictionary
    
    # Get parameters from either JSON body or URL query parameters
    # First check for 'title', then 'q' (which is used in the URL query string from home-search.js)
    movie_title = data.get("title", request.args.get("title", request.args.get("q", "")))
    get_recommendations = data.get("recommend", request.args.get("recommend", "false").lower() == "true")
    
    # Check if this is a genre search
    genre_search = request.args.get("genre", "")
    
    # Write directly to log file for immediate feedback
    with open(log_filename, "a") as log_file:
        if genre_search:
            action_type = "genre search" 
            log_file.write(f"{datetime.datetime.now().isoformat()} - {action_type} for: {genre_search}\n")
        else:
            action_type = "recommendation search" if get_recommendations else "title search"
            log_file.write(f"{datetime.datetime.now().isoformat()} - {action_type} for: {movie_title}\n")
    
    # Log the search
    if genre_search:
        logging.info(f"Genre Search: '{genre_search}'")
    else:
        logging.info(f"Search: '{movie_title}', Get recommendations: {get_recommendations}")
    
    # If recommendation flag is set, use the recommend function
    if get_recommendations:
        # Get number of recommendations requested, default to 10
        try:
            n_recommendations = int(data.get("count", request.args.get("count", 10)))
        except ValueError:
            n_recommendations = 10
            
        recommendations = recommend_movies(movie_title, df, X_scaled, n_recommendations)
        
        # Handle error case
        if isinstance(recommendations, str):
            return jsonify({"error": recommendations}), 404
        
        # Double check for duplicate titles before sending response
        unique_titles = set()
        unique_recommendations = []
        
        # Prepare data for API response
        prepared_recommendations = prepare_movie_data(recommendations)
        recommendation_list = prepared_recommendations.to_dict(orient="records")
        
        # Filter out any remaining duplicates
        for movie in recommendation_list:
            if movie["title"].lower() not in unique_titles:
                unique_titles.add(movie["title"].lower())
                unique_recommendations.append(movie)
        
        # Log if we found duplicates in the final results
        if len(recommendation_list) != len(unique_recommendations):
            logging.warning(f"Removed {len(recommendation_list) - len(unique_recommendations)} duplicate recommendations")
        
        return jsonify({
            "status": "success",
            "query": movie_title,
            "recommendations": unique_recommendations
        })
    
    # If not a recommendation request, search for matching titles
    if not movie_title and not genre_search:
        return jsonify({"error": "No search term provided"}), 400
    
    # If genre_search is provided, search by genre
    if genre_search:
        # Check cache for genre search
        genre_cache_key = f"genre_{genre_search.lower()}"
        if genre_cache_key in recommendation_cache:
            logging.info(f"Using cached results for genre: {genre_search}")
            cached_results = recommendation_cache[genre_cache_key]
            
            # Prepare data for API response
            prepared_results = prepare_movie_data(cached_results)
            results_list = prepared_results.to_dict(orient="records")
            
            return jsonify({
                "status": "success",
                "query": genre_search,
                "total_count": len(cached_results),
                "results": results_list,
                "source": "cache"
            })
        
        logging.info(f"Performing genre search for: {genre_search}")
        
        # Set a timeout for genre search processing
        import time
        start_time = time.time()
        max_processing_time = 15  # Maximum processing time in seconds
        
        # Case-insensitive genre search
        genre_search_lower = genre_search.lower()
        
        # Try to use the genre_index for fast lookups first
        matching_indices = []
        matching_titles = pd.DataFrame()  # Initialize empty DataFrame
        
        # First try exact genre match
        if genre_search_lower in genre_index:
            logging.info(f"Found exact genre match in index for: {genre_search_lower}")
            matching_indices = genre_index[genre_search_lower]
            matching_titles = df.iloc[matching_indices]
        # Then try partial genre matches        else:
            logging.info(f"No exact genre match, trying partial matches for: {genre_search_lower}")
            for g, indices in genre_index.items():
                if genre_search_lower in g:
                    matching_indices.extend(indices)
                    if len(matching_indices) > 1000:  # Limit to avoid too many matches
                        break
            
            # Remove duplicates
            if matching_indices:
                matching_indices = list(set(matching_indices))
                matching_titles = df.iloc[matching_indices]
                
        # If we didn't find matches or found very few, fall back to the old method
        if len(matching_titles) < 20:
            logging.info(f"Few or no matches found with index, using regex search for: {genre_search_lower}")
            try:
                # First try to find movies with the genre as a complete word
                # This is a more precise, faster initial search
                word_boundary_pattern = f"\\b{genre_search_lower}\\b"
                exact_matches = df[df['genres'].str.lower().str.contains(word_boundary_pattern, na=False, regex=True)]
                
                # If we find enough exact matches, use those
                if len(exact_matches) >= 20:
                    matching_titles = exact_matches
                    logging.info(f"Found {len(matching_titles)} exact genre matches for: {genre_search}")
                else:
                    # Fall back to broader search if needed
                    # Find movies matching the genre (partial match)
                    matching_titles = df[df['genres'].str.lower().str.contains(genre_search_lower, na=False)]
                    
                    # If we have a huge number of matches, limit them for performance
                    if len(matching_titles) > 1000:
                        logging.info(f"Found {len(matching_titles)} movies matching genre, limiting results for performance")
                        # If we have ratings, prioritize by rating
                        if 'rating' in matching_titles.columns:
                            matching_titles = matching_titles.sort_values('rating', ascending=False).head(1000)
                        else:
                            # Otherwise just take the first 1000
                            matching_titles = matching_titles.head(1000)
                
                # Check if we're taking too long
                if time.time() - start_time > max_processing_time:
                    logging.warning(f"Genre search timed out after {max_processing_time} seconds, limiting results")
                    if len(matching_titles) > 100:
                        matching_titles = matching_titles.head(100)
                        
            except Exception as e:
                logging.error(f"Error during genre search: {str(e)}")
                # Fall back to a simpler approach in case of error
                matching_titles = df[df['genres'].str.lower().str.contains(genre_search_lower, na=False)].head(100)
        
        # Log the search results
        logging.info(f"Returning {len(matching_titles)} movies matching genre: {genre_search}")
    else:        # Search for partial matches in titles (case-insensitive)
        search_term = movie_title.lower()
        matching_titles = df[df['title'].str.lower().str.contains(search_term, na=False)]
        
        # If no results found in titles, try searching in genres as a fallback
        if len(matching_titles) == 0:
            logging.info(f"No title matches for '{search_term}', trying genre search")
            matching_titles = df[df['genres'].str.lower().str.contains(search_term, na=False)]
        
        # Limit results for performance
        if len(matching_titles) > 1000:
            logging.info(f"Found {len(matching_titles)} title matches, limiting results for performance")
            # If we have ratings, prioritize by rating
            if 'rating' in matching_titles.columns:
                matching_titles = matching_titles.sort_values('rating', ascending=False).head(1000)
            else:
                # Otherwise just take the first 1000
                matching_titles = matching_titles.head(1000)
    
    # Get total count for pagination info
    total_count = len(matching_titles)
    
    # Sort by rating if available, otherwise by title
    if 'rating' in matching_titles.columns:
        matching_titles = matching_titles.sort_values('rating', ascending=False)
    else:
        matching_titles = matching_titles.sort_values('title')
    
    # Take only a reasonable number of results for the response
    # This ensures we don't return massive amounts of data
    # Allow for 2 pages (40 results)
    matching_titles = matching_titles.head(40)
    
    # Filter out duplicate titles (case insensitive)
    unique_titles = set()
    unique_results = pd.DataFrame()
    
    for _, movie in matching_titles.iterrows():
        title_lower = movie['title'].lower() if isinstance(movie['title'], str) else ''
        if title_lower not in unique_titles:
            unique_titles.add(title_lower)
            unique_results = pd.concat([unique_results, movie.to_frame().T])
    
    # Take up to 40 results (for 2 pages)
    results = unique_results.head(40)
    
    # Check if we have any results
    if len(results) == 0:
        return jsonify({
            "status": "no_results",
            "query": movie_title or genre_search,
            "total_count": 0,
            "results": []
        })
    
    # Prepare data for API response
    prepared_results = prepare_movie_data(results)
    results_list = prepared_results.to_dict(orient="records")
    
    # Cache genre search results if this was a genre search
    if genre_search and results_list:
        genre_cache_key = f"genre_{genre_search.lower()}"
        recommendation_cache[genre_cache_key] = results
        logging.info(f"Cached results for genre: {genre_search}")
    
    return jsonify({
        "status": "success",
        "query": movie_title or genre_search,
        "total_count": total_count,
        "results": results_list
    })

@app.route("/search-results.html")
def search_results_html():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "search-results.html")

@app.route("/search.js")
def search_js():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "search.js")

@app.route("/search-results-fixed.html")
def search_results_fixed_html():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "search-results-fixed.html")

@app.route("/quick_test.html")
def quick_test_html():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "quick_test.html")

@app.route("/test_recommendations.html")
def test_recommendations_html():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "test_recommendations.html")

@app.route("/recommend", methods=["POST", "GET"])
@jsonpify
def recommend():
    """Endpoint to get movie recommendations"""
    # Handle both POST and GET requests
    if request.method == "POST":
        try:
            data = request.get_json(silent=True) or {}
        except Exception as e:
            data = {}
            logging.error(f"Error parsing JSON data: {str(e)}")
    else:  # GET request
        data = {}  # Initialize as empty dictionary
    
    # Get title from either JSON body or URL query parameters
    movie_title = data.get("title", request.args.get("title", ""))
    
    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400
    
    # Get number of recommendations requested, default to 10
    try:
        n_recommendations = int(data.get("count", request.args.get("count", 10)))
    except ValueError:
        n_recommendations = 10
    
    recommendations = recommend_movies(movie_title, df, X_scaled, n_recommendations)
    
    # Handle error case
    if isinstance(recommendations, str):
        return jsonify({"error": recommendations}), 404
    
    # Double check for duplicate titles before sending response
    unique_titles = set()
    unique_recommendations = []
    
    # Prepare data for API response
    prepared_recommendations = prepare_movie_data(recommendations)
    results = prepared_recommendations.to_dict(orient="records")
    
    # Filter out any remaining duplicates
    for movie in results:
        if movie["title"].lower() not in unique_titles:
            unique_titles.add(movie["title"].lower())
            unique_recommendations.append(movie)
    
    # Log if we found duplicates in the final results
    if len(results) != len(unique_recommendations):
        logging.warning(f"Removed {len(results) - len(unique_recommendations)} duplicate recommendations")
    
    return jsonify({
        "status": "success",
        "query": movie_title,
        "recommendations": unique_recommendations
    })

@app.route("/movie-details", methods=["POST", "GET"])
@jsonpify
def movie_details():
    """Endpoint to get detailed information about a movie"""
    # Get the movie title from either JSON body or URL query parameter
    if request.method == "POST":
        try:
            data = request.get_json(silent=True) or {}
            movie_title = data.get("title", "")
        except Exception as e:
            movie_title = ""
            logging.error(f"Error parsing JSON data: {str(e)}")
    else:  # GET request
        movie_title = request.args.get("title", "")
    
    if movie_title:
        # Log the redirect
        logging.info(f"Redirecting movie details request for: {movie_title} to search-results")
        # Instead of detailed movie info, redirect to search results with recommendation mode
        from flask import redirect
        return redirect(f"/search-results.html?q={movie_title}&recommend=true")
    
    # If no title provided, return an error message as JSON
    return jsonify({
        "status": "error",
        "message": "Movie details page has been removed. Please use search page instead."
    })

@app.route("/movie-details.html")
def movie_details_html():
    # Redirect to home page
    from flask import redirect
    title = request.args.get("title", "")
    if title:
        return redirect(f"/search-results.html?q={title}&recommend=true")
    return redirect("/")

@app.route("/performance_dashboard.html")
def performance_dashboard_html():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "performance_dashboard.html")

@app.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint to get detailed information about the system state"""
    try:
        import sys
        import numpy as np
        
        # Get dataset info
        dataset_info = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "has_clusters": 'cluster' in df.columns,
            "cluster_distribution": df['cluster'].value_counts().to_dict() if 'cluster' in df.columns else {},
            "columns": list(df.columns)
        }
        
        # Get system info
        system_info = {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "current_time": datetime.datetime.now().isoformat()
        }
        
        return jsonify({
            "status": "success",
            "dataset_info": dataset_info,
            "system_info": system_info
        })
    except Exception as e:
        logging.error(f"Error in debug endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route("/performance-stats", methods=["GET"])
def performance_stats():
    """Endpoint to get performance statistics for the dashboard"""
    try:
        # Get cache statistics
        cache_stats = {
            "cache_size": len(recommendation_cache),
            "cached_movies": list(recommendation_cache.keys())[:10]  # Just show first 10 to keep response small
        }
        
        # Get recent movie details requests from log
        recent_requests = []
        try:
            with open(log_filename, "r") as log_file:
                lines = log_file.readlines()
                # Get last 20 lines that contain "processed in" (our performance timing logs)
                perf_lines = [line for line in lines if "processed in" in line][-20:]
                for line in perf_lines:
                    parts = line.split("processed in")
                    if len(parts) == 2:
                        movie_part = parts[0].split("for:")
                        if len(movie_part) == 2:
                            movie_title = movie_part[1].strip().strip("'")
                            time_part = parts[1].strip()
                            recent_requests.append({
                                "movie": movie_title,
                                "processing_time": time_part
                            })
        except Exception as log_error:
            logging.error(f"Error reading log file: {str(log_error)}")
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.datetime.now().isoformat(),
            "cache_stats": cache_stats,
            "recent_requests": recent_requests
        })
    except Exception as e:
        logging.error(f"Error in performance stats endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route("/movies-by-genre")
@jsonpify
def movies_by_genre():
    """Return movies filtered by genre"""
    start_time = datetime.datetime.now()
    
    # Get parameters
    genre = request.args.get("genre", "")
    limit_str = request.args.get("limit", "10")
    try:
        limit = int(limit_str)
    except ValueError:
        limit = 10
    
    # Log the request
    logging.info(f"Movies by genre request: genre={genre}, limit={limit}")
    
    if not genre:
        return jsonify({
            "error": "No genre specified",
            "processing_time_seconds": (datetime.datetime.now() - start_time).total_seconds()
        })
    
    try:
        # Filter movies by genre
        # Create a case-insensitive search for the genre in the genres column
        # Special handling for Sci-Fi which might be stored as "Sci-Fi" or "SciFi"
        if genre.lower() == 'sci-fi':
            filtered_movies = df[df['genres'].str.lower().str.contains('sci-fi', case=False, na=False) | 
                                df['genres'].str.lower().str.contains('scifi', case=False, na=False)]
        else:
            filtered_movies = df[df['genres'].str.contains(genre, case=False, na=False)]
        
        # If no movies found for this genre, return some popular movies as fallback
        is_fallback = False
        if len(filtered_movies) == 0:
            logging.warning(f"No movies found for genre: {genre}. Using popular movies as fallback.")
            if 'rating' in df.columns:
                filtered_movies = df.sort_values('rating', ascending=False).head(6)
            else:
                filtered_movies = df.head(6)
            is_fallback = True
        
        # Sort by rating (descending)
        if 'rating' in filtered_movies.columns:
            filtered_movies = filtered_movies.sort_values('rating', ascending=False)
        else:
            filtered_movies = filtered_movies.sort_values('title')
        
        # Get total count before limiting
        total_count = len(filtered_movies)
        
        # Limit results
        filtered_movies = filtered_movies.head(limit)
        
        # Convert to list of dictionaries
        movies_list = []
        for _, movie in filtered_movies.iterrows():
            movie_dict = {
                "title": movie['title'],
                "genres": movie['genres']
            }
            
            if 'year' in movie and not pd.isna(movie['year']):
                movie_dict["year"] = str(int(movie['year']))
            
            if 'rating' in movie and not pd.isna(movie['rating']):
                movie_dict["rating"] = float(movie['rating'])
                
            movies_list.append(movie_dict)
        
        # Calculate processing time
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Return results
        return jsonify({
            "movies": movies_list,
            "total_count": total_count,
            "genre": genre,
            "is_fallback": is_fallback,
            "processing_time_seconds": processing_time
        })
    
    except Exception as e:
        logging.error(f"Error in movies-by-genre endpoint: {str(e)}")
        return jsonify({
            "error": f"Error processing request: {str(e)}",
            "processing_time_seconds": (datetime.datetime.now() - start_time).total_seconds()
        })

@app.route("/discover.js")
def discover_js():
    # First try to serve from static folder
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    if os.path.exists(os.path.join(static_path, "discover.js")):
        return send_from_directory(static_path, "discover.js")
    else:
        # Fall back to root directory
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "discover.js")

@app.route("/discover.html")
def discover_html():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "discover.html")

@app.route("/popular-movies")
def popular_movies():
    """Endpoint to get popular movies"""
    try:
        # Sort by rating if available
        if 'rating' in df.columns:
            popular = df.sort_values(by='rating', ascending=False).head(10)
        else:
            # Otherwise just return the first 10
            popular = df.head(10)
        
        # Prepare data for API response
        available_columns = ['movieId', 'title', 'genres', 'year', 'rating', 'poster_url']
        available_columns = [col for col in available_columns if col in popular.columns]
        
        # Convert DataFrame to dictionary for JSON serialization
        results_list = popular[available_columns].to_dict(orient="records")
        
        return jsonify({
            "status": "success",
            "results": results_list
        })
    except Exception as e:
        logging.error(f"Error in popular-movies: {str(e)}")
        return jsonify({"error": str(e), "status": "error"})

# Generic route to serve any HTML file from the root directory
@app.route("/<path:filename>.html")
def serve_html(filename):
    """Serve any HTML file from the root directory"""
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{filename}.html")
    if os.path.exists(full_path):
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), f"{filename}.html")
    else:
        return f"File {filename}.html not found", 404

@app.errorhandler(404)
def page_not_found(e):
    """Redirect users to home page for any missing pages"""
    # Check if this is a movie details request
    path = request.path
    if "movie-details" in path.lower():
        # Extract title from query parameter if it exists
        title = request.args.get("title", "")
        if title:
            return redirect(f"/search-results.html?q={title}&recommend=true")
        else:
            return redirect("/")
    # For other 404 errors, return standard error page
    return f"Page not found: {path}", 404

def get_additional_recommendations(recommendations, movie, original_genres, n_recommendations, df, seen_titles):
    # Calculate how many more recommendations we need
    remaining = n_recommendations - len(recommendations)
    if remaining <= 0:
        return recommendations
    logging.info(f"Finding {remaining} additional movies with similar genres")
    similar_genre_movies = pd.DataFrame()
    if original_genres and len(original_genres) > 0:
        genre_filter = None
        for genre in original_genres:
            condition = df['genres'].astype(str).str.contains(genre, na=False)
            if genre_filter is None:
                genre_filter = condition
            else:
                genre_filter = genre_filter | condition
        if genre_filter is not None:
            filtered_movies = df[genre_filter].copy()
            mask = ~filtered_movies['title'].astype(str).str.lower().isin([t.lower() for t in seen_titles])
            filtered_movies = filtered_movies[mask]
            if isinstance(movie['title'], str):
                mask = filtered_movies['title'].astype(str).str.lower() != movie['title'].lower()
                filtered_movies = filtered_movies[mask]
            if 'rating' in filtered_movies.columns:
                filtered_movies = filtered_movies.sort_values('rating', ascending=False)
            similar_genre_movies = filtered_movies.head(remaining)
            for title in similar_genre_movies['title'].astype(str):
                seen_titles.add(title.lower())
    if len(similar_genre_movies) > 0:
        recommendations = pd.concat([recommendations, similar_genre_movies])
        remaining = n_recommendations - len(recommendations)
    if remaining > 0:
        logging.info(f"Adding {remaining} random popular movies")
        other_movies = df[~df['title'].isin(recommendations['title'])]
        other_movies = other_movies[other_movies['title'] != movie['title']]
        filtered_movies = pd.DataFrame()
        for _, row in other_movies.iterrows():
            rec_title_lower = row['title'].lower() if isinstance(row['title'], str) else ''
            if rec_title_lower not in seen_titles:
                filtered_movies = pd.concat([filtered_movies, row.to_frame().T])
                seen_titles.add(rec_title_lower)
                if len(filtered_movies) >= remaining:
                    break
        if 'rating' in filtered_movies and not filtered_movies.empty:
            filtered_movies = filtered_movies.sort_values('rating', ascending=False)
        if not filtered_movies.empty:
            recommendations = pd.concat([recommendations, filtered_movies.head(remaining)])
    return recommendations

if __name__ == "__main__":
    # Always use port 3000
    try:
        logging.info(f"Starting server on port {SERVER_PORT}")
        print(f"Starting server on port {SERVER_PORT}...")
        app.run(host="0.0.0.0", port=SERVER_PORT, debug=False, threaded=True)
    except Exception as port_error:
        logging.critical(f"Could not start server on port {SERVER_PORT}: {str(port_error)}")
        print(f"Error: Port {SERVER_PORT} is in use. Please close any applications using this port and try again.")

