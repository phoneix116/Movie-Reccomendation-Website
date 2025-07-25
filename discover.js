// Discover page functionality
document.addEventListener('DOMContentLoaded', function() {
    // Get references to genre movie containers
    const actionMoviesContainer = document.getElementById('action-movies');
    const comedyMoviesContainer = document.getElementById('comedy-movies');
    const dramaMoviesContainer = document.getElementById('drama-movies');
    const scifiMoviesContainer = document.getElementById('scifi-movies');
    const romanceMoviesContainer = document.getElementById('romance-movies');
    const animationMoviesContainer = document.getElementById('animation-movies');

    // Get the current host and port from window.location
    const baseUrl = window.location.origin;    // Load movies by genre
    loadMoviesByGenre('Action', actionMoviesContainer);
    loadMoviesByGenre('Comedy', comedyMoviesContainer);
    loadMoviesByGenre('Drama', dramaMoviesContainer);
    loadMoviesByGenre('Sci-Fi', scifiMoviesContainer);
    loadMoviesByGenre('Romance', romanceMoviesContainer);
    loadMoviesByGenre('Animation', animationMoviesContainer);
    
    // Add recommendation button functionality
    const recommendationBtn = document.querySelector('.recommendation-btn');
    const recommendationModal = document.getElementById('recommendationModal');
    const closeModalBtn = document.querySelector('.close-modal');
    
    if (recommendationBtn && recommendationModal) {
        recommendationBtn.addEventListener('click', function(e) {
            e.preventDefault();
            recommendationModal.style.display = 'block';
            const recommendationInput = document.getElementById('recommendationInput');
            if (recommendationInput) recommendationInput.focus();
        });
    }

    // Function to load movies by genre
    function loadMoviesByGenre(genre, container) {
        if (!container) return;

        console.log(`Loading ${genre} movies...`);
        
        // Add a timestamp to prevent caching
        const timestamp = new Date().getTime();
        
        fetch(`${baseUrl}/movies-by-genre?genre=${encodeURIComponent(genre)}&limit=6&_=${timestamp}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {            console.log(`Loaded ${data.movies ? data.movies.length : 0} ${genre} movies`);
            
            if (data.error) {
                container.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                return;
            }
              // Check if we're showing fallback movies
            if (data.is_fallback) {
                console.log(`Using fallback popular movies for ${genre}`);
                
                // Add a notification about fallback movies
                const fallbackNotice = document.createElement('div');
                fallbackNotice.className = 'fallback-notice';
                fallbackNotice.innerHTML = `No ${genre} movies found. Showing popular movies instead.`;
                container.appendChild(fallbackNotice);
            }
              // Check if we have any results
            if (!data.movies || data.movies.length === 0) {
                container.innerHTML = `
                    <div class="no-results">
                        <p>No ${genre} movies found</p>
                        <button class="retry-btn" data-genre="${genre}">Try Again</button>
                    </div>`;
                    
                // Add event listener to the retry button
                const retryBtn = container.querySelector('.retry-btn');
                if (retryBtn) {
                    retryBtn.addEventListener('click', function() {
                        const genre = this.getAttribute('data-genre');
                        container.innerHTML = `<div class="loading">Loading ${genre} movies...</div>`;
                        loadMoviesByGenre(genre, container);
                    });
                }
                return;
            }
            
            // Clear the container
            container.innerHTML = '';
            
            // Create cards for each movie
            data.movies.forEach(movie => {
                const card = createMovieCard(movie);
                container.appendChild(card);
            });
              // Add a "See More" button if there are more than 6 movies
            if (data.total_count && data.total_count > data.movies.length) {
                const seeMoreBtn = document.createElement('div');
                seeMoreBtn.className = 'see-more-btn';
                seeMoreBtn.innerHTML = `<a href="/search-results.html?q=${encodeURIComponent(genre)}&recommend=true">See More ${genre} Movies</a>`;
                container.appendChild(seeMoreBtn);
                
                // Add click event to search for more movies of this genre
                const seeMoreLink = seeMoreBtn.querySelector('a');
                if (seeMoreLink) {
                    seeMoreLink.addEventListener('click', function(e) {
                        // Prevent default navigation
                        e.preventDefault();
                        
                        // Add a timestamp to prevent caching
                        const timestamp = new Date().getTime();
                        
                        // Navigate to search results with genre as search term
                        window.location.href = `/search-results.html?q=${encodeURIComponent(genre)}&recommend=true&_=${timestamp}`;
                    });
                }
            }
        })
        .catch(error => {
            console.error(`Error loading ${genre} movies:`, error);
            container.innerHTML = `<div class="error">Failed to load ${genre} movies: ${error.message}</div>`;
        });
    }

    // Function to create a movie card
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
                <a href="/search-results.html?q=${encodeURIComponent(title)}&recommend=true">
                    <div class="movie-info">
                        <div class="movie-title">${title}</div>
                        <div class="movie-year">${year}</div>
                        <div class="movie-genres">${genres}</div>
                        ${ratingDisplay}
                    </div>
                </a>
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
        
        return card;
    }

    // Handle recommendation button (using the one declared above)
    if (recommendationBtn) {
        // This is a backup handler in case the first one didn't work
        recommendationBtn.addEventListener('click', function(e) {
            e.preventDefault();
            const recommendationModal = document.getElementById('recommendationModal');
            if (recommendationModal) {
                recommendationModal.style.display = 'block';
                const recommendationInput = document.getElementById('recommendationInput');
                if (recommendationInput) recommendationInput.focus();
            }
        });
    }

    // Handle modal close (using the closeModalBtn declared above)
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
        }    });
    
    // Handle recommendation input and button in the modal
    // Define these variables if they don't already exist
    let recommendationInput = document.getElementById('recommendationInput');
    let recommendationSearchButton = document.getElementById('recommendationSearchButton');
    let popularMovieButtons = document.querySelectorAll('.popular-movie');
    
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
    if (popularMovieButtons.length > 0) {
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
});


