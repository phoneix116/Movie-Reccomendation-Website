// Handle movie search and recommendation functionality
document.addEventListener('DOMContentLoaded', function() {
    // Get references to DOM elements
    const searchInput = document.querySelector('.search-box input');
    const searchButton = document.querySelector('.search-box button');
    const recommendationBtn = document.querySelector('.recommendation-btn');
    const movieGrid = document.querySelector('.movie-grid');
    
    // Skip Firebase entirely and use Flask backend directly
    console.log("Using Flask backend directly for all searches");
    
    // Search button functionality - only apply if the element exists
    if (searchButton) {
        searchButton.addEventListener('click', function() {
            const searchTerm = searchInput.value.trim();
            if (searchTerm) {
                // Redirect to the search results page
                window.location.href = `/search-results.html?q=${encodeURIComponent(searchTerm)}`;
            }
        });
    }

    // Enter key in search field - only apply if the element exists
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                const searchTerm = searchInput.value.trim();
                if (searchTerm) {
                    // Redirect to the search results page
                    window.location.href = `/search-results.html?q=${encodeURIComponent(searchTerm)}`;
                }
            }
        });
    }
    
    // Recommendation button functionality - use modal instead of prompt
    if (recommendationBtn) {
        recommendationBtn.addEventListener('click', function(e) {
            e.preventDefault();
            // Show the modal if it exists, otherwise fallback to search results
            const recommendationModal = document.getElementById('recommendationModal');
            if (recommendationModal) {
                recommendationModal.style.display = 'block';
                const recommendationInput = document.getElementById('recommendationInput');
                if (recommendationInput) recommendationInput.focus();
            }
        });
    }
    
    // Add listeners for the close modal button and outside clicks if they don't exist yet
    const closeModalBtn = document.querySelector('.close-modal');
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', function() {
            const recommendationModal = document.getElementById('recommendationModal');
            if (recommendationModal) {
                recommendationModal.style.display = 'none';
            }
        });
    }
    
    // Close modal when clicking outside of it
    window.addEventListener('click', function(e) {
        const recommendationModal = document.getElementById('recommendationModal');
        if (e.target === recommendationModal) {
            recommendationModal.style.display = 'none';
        }
    });
    
    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        const recommendationModal = document.getElementById('recommendationModal');
        if (e.key === 'Escape' && recommendationModal && recommendationModal.style.display === 'block') {
            recommendationModal.style.display = 'none';
        }
    });
    
    // Handle recommendation input and button in the modal
    const recommendationInput = document.getElementById('recommendationInput');
    const recommendationSearchButton = document.getElementById('recommendationSearchButton');
    const popularMovieButtons = document.querySelectorAll('.popular-movie');
    
    if (recommendationInput && recommendationSearchButton) {
        // Handle recommendation search button click
        recommendationSearchButton.addEventListener('click', function() {
            const movieTitle = recommendationInput.value.trim();
            if (movieTitle) {
                // Show loading state
                this.classList.add('loading');
                this.textContent = 'Finding...';
                
                // Add a timestamp to prevent caching
                const timestamp = new Date().getTime();
                setTimeout(() => {
                    window.location.href = `search-results.html?q=${encodeURIComponent(movieTitle)}&recommend=true&_=${timestamp}`;
                }, 300);
            }
        });
        
        // Handle "Enter" key in recommendation input
        recommendationInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                const movieTitle = recommendationInput.value.trim();
                if (movieTitle) {
                    // Show loading state on the button
                    if (recommendationSearchButton) {
                        recommendationSearchButton.classList.add('loading');
                        recommendationSearchButton.textContent = 'Finding...';
                    }
                    
                    // Add a timestamp to prevent caching
                    const timestamp = new Date().getTime();
                    setTimeout(() => {
                        window.location.href = `search-results.html?q=${encodeURIComponent(movieTitle)}&recommend=true&_=${timestamp}`;
                    }, 300);
                }
            }
        });
    }
    
    // Handle popular movie buttons in the modal
    if (popularMovieButtons && popularMovieButtons.length > 0) {
        popularMovieButtons.forEach(button => {
            button.addEventListener('click', function() {
                const movieTitle = this.dataset.title;
                if (movieTitle) {
                    // Add visual feedback with class
                    this.classList.add('popular-movie-clicked');
                    
                    // Small delay before navigation to show the animation
                    setTimeout(() => {
                        // Add a timestamp to prevent caching
                        const timestamp = new Date().getTime();
                        window.location.href = `search-results.html?q=${encodeURIComponent(movieTitle)}&recommend=true&_=${timestamp}`;
                    }, 200);
                }
            });
        });
    }

    // Function to search directly in Flask backend
    function searchDirectInDB(searchTerm) {
        if (!movieGrid) return;

        console.log("Searching directly in Flask backend for:", searchTerm);
        
        // Show loading state
        movieGrid.innerHTML = '<div class="loading">Searching for movies...</div>';
        
        fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title: searchTerm }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Search response:", data);
            
            if (data.error) {
                console.warn("Search error:", data.error);
                // If search fails, fall back to recommendations
                getRecommendations(searchTerm);
                return;
            }
            
            const results = data.results;
            if (results && results.length > 0) {
                console.log(`Found ${results.length} results in Flask database`);
                
                // Clear the movie grid
                movieGrid.innerHTML = '';
                
                // Create a movie card for each result
                results.forEach(movie => {
                    const card = createMovieCard(movie);
                    movieGrid.appendChild(card);
                });
            } else {
                console.log("No results found in search, trying recommendations");
                // If no search results, try recommendations
                getRecommendations(searchTerm);
            }
        })
        .catch(error => {
            console.error('Error searching movies:', error);
            // Fall back to recommendations if search fails
            getRecommendations(searchTerm);
        });
    }
    
    // Function to get recommendations from the Flask backend
    function getRecommendations(movieTitle) {
        if (!movieGrid) return;

        console.log("Getting recommendations for:", movieTitle);
        
        // Show loading state
        movieGrid.innerHTML = '<div class="loading">Loading recommendations...</div>';
        
        fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title: movieTitle }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Recommendation response:", data);
            
            if (data.error) {
                console.warn("Recommendation error:", data.error);
                movieGrid.innerHTML = `<div class="error">${data.error}</div>`;
                return;
            }
            
            const recommendations = data.recommendations;
            if (recommendations && recommendations.length > 0) {
                console.log(`Found ${recommendations.length} recommendations`);
                
                // Clear the movie grid
                movieGrid.innerHTML = '';
                
                // Create a movie card for each recommendation
                recommendations.forEach(movie => {
                    try {
                        const card = createMovieCard(movie);
                        movieGrid.appendChild(card);
                    } catch (cardError) {
                        console.error("Error creating movie card:", cardError, movie);
                    }
                });
            } else {
                console.warn("No recommendations found");
                movieGrid.innerHTML = '<div class="error">No recommendations found</div>';
            }
        })
        .catch(error => {
            console.error('Error fetching recommendations:', error);
            movieGrid.innerHTML = `<div class="error">Error fetching recommendations: ${error.message}</div>`;
        });
    }
    
    // Function to create a movie card element
    function createMovieCard(movie) {
        const card = document.createElement('div');
        card.className = 'movie-card';
        
        try {
            // Format the rating with one decimal place if it exists
            let ratingDisplay = '';
            if (movie.rating !== undefined && movie.rating !== null) {
                try {
                    const ratingValue = typeof movie.rating === 'number' ? movie.rating : parseFloat(movie.rating);
                    if (!isNaN(ratingValue)) {
                        ratingDisplay = `<div class="movie-rating">Rating: ${ratingValue.toFixed(1)}</div>`;
                    } else {
                        ratingDisplay = `<div class="movie-rating">Rating: N/A</div>`;
                    }
                } catch (e) {
                    console.error("Error formatting rating:", e, movie.rating);
                    ratingDisplay = `<div class="movie-rating">Rating: ${movie.rating}</div>`;
                }
            }
            
            const title = movie.title || 'Unknown Title';
            const year = movie.year || 'N/A';
            const genres = movie.genres || movie.genre || 'N/A';
            
            card.innerHTML = `
                <div class="movie-info">
                    <div class="movie-title">${title}</div>
                    <div class="movie-year">${year}</div>
                    <div class="movie-genres">${genres}</div>
                    ${ratingDisplay}
                    <button class="recommend-similar" data-title="${title}">Similar Movies</button>
                </div>
            `;
            
        } catch (error) {
            console.error("Error creating movie card:", error, movie);
            card.innerHTML = `
                <div class="movie-info">
                    <div class="movie-title">${movie.title || 'Unknown Title'}</div>
                    <div class="error">Error displaying movie details</div>
                </div>
            `;
        }
        
        // Add event listener for the similar movies button
        const similarButton = card.querySelector('.recommend-similar');
        if (similarButton) {
            similarButton.addEventListener('click', function() {
                const movieTitle = this.getAttribute('data-title');
                getRecommendations(movieTitle);
            });
        }
        
        return card;
    }
});
