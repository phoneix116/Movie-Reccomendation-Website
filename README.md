# Movie Recommendation System

A full-stack movie recommendation web application with content-based and cluster-based recommendation algorithms.

![Movie Recommendation System](https://img.shields.io/badge/Status-Active-brightgreen)
![Version](https://img.shields.io/badge/Version-1.1.0-blue)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Overview

This project is a movie recommendation system that suggests movies based on user preferences, movie similarity, and genre preferences. The system uses content-based filtering and clustering techniques to provide personalized recommendations from a dataset of over 25 million movie entries.

## ✨ Features

- **Search Functionality**: Search movies by title or partial title match
- **Genre-Based Search**: Browse movies by specific genres
- **Content-Based Recommendations**: Get movies similar to ones you like based on genres and features
- **Cluster-Based Recommendations**: Alternative recommendation method using KMeans clustering
- **Caching System**: Improved performance with cached results for repeat searches
- **Responsive Web Interface**: Clean and user-friendly web interface
- **Performance Optimizations**: Efficient algorithms to handle large datasets
- **Activity Logging**: Comprehensive logging for monitoring and debugging

## 🔄 Recent Updates

### Performance Optimizations
- **Persistent Caching**: Added file-based caching for processed dataframe (`df_cache.pkl`) and lookup indices (`indices_cache.pkl`)
- **Fast Lookup Indices**: Implemented multi-tier search with specialized indices for titles, titles without years, and word matching
- **Batch Processing**: Added batch processing for index building to reduce memory usage with large datasets
- **Annoy Integration**: Added approximate nearest-neighbor search with lazy loading and timeout protection

### Feature Flags
- **ENABLE_ANNOY**: Toggle expensive Annoy-based recommendation features
- **ENABLE_CONTENT_BASED**: Control content-based recommendation algorithms
- **FAST_STARTUP**: Run in development mode with `--fast` flag to load only a subset of data

### Search Improvements
- **Multi-Strategy Search**: Enhanced title matching with hierarchical fallback strategies
- **Smart Caching**: Improved in-memory caching for recommendations and genre searches
- **Fallback Mechanisms**: Graceful degradation with timeout protection and multiple fallback methods

### API Enhancements
- **CORS Support**: Added proper cross-origin resource sharing support
- **JSONP Support**: Added JSONP response wrapping via a decorator
- **Enhanced Error Handling**: Comprehensive error recovery with detailed logging

## 🛠️ Technologies Used

### Backend
- **Python 3.10**: Core programming language
- **Flask**: Web server framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **scikit-learn**: Machine learning algorithms for clustering and recommendations
- **Flask-CORS**: Cross-Origin Resource Sharing support

### Frontend
- **HTML/CSS/JavaScript**: Frontend development
- **Responsive Design**: Works on desktop and mobile devices

## 📊 Dataset

The system uses three main datasets:
- `df.csv` (~3GB): Main dataset with movie information and pre-computed features
- `movies.csv` (~3MB): Basic movie information
- `ratings.csv` (~678MB): User ratings data

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Sufficient RAM (16GB+ recommended for optimal performance)
- Git

### Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system
```

> **Note:** When publishing to GitHub, replace `YOUR_GITHUB_USERNAME` with your actual GitHub username (e.g., `johnsmith`). People cloning your repository will use this updated URL.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
python backend_df_direct_fixed.py
```

For faster startup during development (loads a smaller dataset subset):
```bash
python backend_df_direct_fixed.py --fast
```

The application will be available at http://localhost:3000

> **Note:** On first run, the system will process the dataset and build necessary indices, which may take some time. Cache files will be automatically generated to speed up subsequent startups.

## 📝 Usage

1. **Home Page**: Enter a movie title in the search box or browse by genre
2. **Search Results**: View matching movies and select one for recommendations
3. **Recommendation Page**: Get personalized movie recommendations based on your selection
4. **Discover**: Explore popular movies and trending genres

## 🔍 API Endpoints

- `/search`: Search for movies by title or get recommendations
- `/recommend`: Get movie recommendations based on a title
- `/movie-details`: Get detailed information about a movie
- `/api-status`: Check API status
- `/clear-cache`: Clear the recommendation cache

## ⚙️ Performance Considerations

- The application uses advanced caching techniques to improve response times
- Timeout mechanisms prevent long-running operations
- Vectorized operations and efficient filtering for large dataset handling
- Boolean indexing and sampling techniques for improved performance

## 📁 Project Structure

```
├── backend_df_direct_fixed.py   # Main Flask application
├── df.csv                       # Main dataset
├── movies.csv                   # Movie information
├── ratings.csv                  # User ratings
├── requirements.txt             # Python dependencies
├── index.html                   # Home page
├── search-results.html          # Search results page
├── styles.css                   # Main CSS styles
├── static/                      # Static assets
│   ├── css/                     # CSS files
│   └── js/                      # JavaScript files
└── activity_log_*.log           # Activity logs
```

> **Note:** Cache files (`df_cache.pkl` and `indices_cache.pkl`) will be generated automatically when you run the application for the first time. These files significantly improve startup time on subsequent runs and should not be committed to version control.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Kaggle](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system) - For the dataset
- [scikit-learn](https://scikit-learn.org/) - For machine learning algorithms
- [Flask](https://flask.palletsprojects.com/) - For the web framework



Project Link: [https://github.com/YOUR_GITHUB_USERNAME/movie-recommendation-system](https://github.com/YOUR_GITHUB_USERNAME/movie-recommendation-system)

> **Note:** When publishing to GitHub, replace `YOUR_GITHUB_USERNAME` with your actual GitHub username.
