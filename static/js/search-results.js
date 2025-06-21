// Search Results Page Script
document.addEventListener('DOMContentLoaded', function() {
    // Page elements
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');
    const searchQueryDisplay = document.getElementById('search-query');
    const searchResults = document.getElementById('searchResults');
    const noResults = document.querySelector('.no-results');
    const sortBySelect = document.getElementById('sort-by');
    const filterBtn = document.getElementById('filter-btn');
    const prevPageBtn = document.getElementById('prev-page');
    const nextPageBtn = document.getElementById('next-page');
    const pageInfo = document.getElementById('page-info');
    
    // Variables to track pagination and results
    let currentPage = 1;
    let resultsPerPage = 12;
    let totalResults = 0;
    let currentResults = [];
    let currentQuery = '';
    
    // Get the search query from URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    const searchTerm = urlParams.get('q');
    const isRecommend = urlParams.get('recommend') === 'true';
    
    // Don't check API if it's already been checked by the inline script
    if (!window.serverConnectionChecked) {
        checkAPIAvailability();
    } else {
        console.log("Server connection already checked by inline script");
        // Create an API ready event to trigger search
        const apiReadyEvent = new CustomEvent('apiReady', { detail: { port: 3000 } });
        document.dispatchEvent(apiReadyEvent);
    }
    
    // Listen for API ready event from server connection check
    document.addEventListener('apiReady', function(e) {
        console.log('API is ready on port:', e.detail.port);
        
        // Execute search if there's a search term in URL
        if (searchTerm) {
            if (isRecommend) {
                getRecommendations(searchTerm);
            } else {
                performSearch(searchTerm);
            }
        }
    });
    
    if (searchTerm) {
        // Update UI
        searchInput.value = searchTerm;
        searchQueryDisplay.textContent = isRecommend ? 
            `Recommendations similar to: ${searchTerm}` : searchTerm;
        currentQuery = searchTerm;
        
        // Search will be performed when API is ready via the apiReady event
    }
    
    // Search button click handler
    searchButton.addEventListener('click', function() {
        const newSearchTerm = searchInput.value.trim();
        if (newSearchTerm) {
            // Update URL with new search term
            const newUrl = new URL(window.location);
            newUrl.searchParams.set('q', newSearchTerm);
            newUrl.searchParams.delete('recommend');
            window.history.pushState({}, '', newUrl);
            
            searchQueryDisplay.textContent = newSearchTerm;
            currentQuery = newSearchTerm;
            currentPage = 1;
            performSearch(newSearchTerm);
        }
    });
    
    // Search on Enter key press
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            const newSearchTerm = searchInput.value.trim();
            if (newSearchTerm) {
                // Update URL with new search term
                const newUrl = new URL(window.location);
                newUrl.searchParams.set('q', newSearchTerm);
                newUrl.searchParams.delete('recommend');
                window.history.pushState({}, '', newUrl);
                
                searchQueryDisplay.textContent = newSearchTerm;
                currentQuery = newSearchTerm;
                currentPage = 1;
                performSearch(newSearchTerm);
            }
        }
    });
    
    // Filter button click handler
    filterBtn.addEventListener('click', function() {
        applyFilters();
    });
    
    // Make sure dropdown always uses dark styling
    sortBySelect.addEventListener('focus', function() {
        // Add a class to allow custom styling when dropdown is open
        this.classList.add('active');
    });
    
    sortBySelect.addEventListener('blur', function() {
        // Remove class when dropdown is closed
        this.classList.remove('active');
    });
    
    // Pagination button click handlers
    prevPageBtn.addEventListener('click', function() {
        if (currentPage > 1) {
            currentPage--;
            displayResults(currentResults);
        }
    });
    
    nextPageBtn.addEventListener('click', function() {
        const maxPage = Math.ceil(totalResults / resultsPerPage);
        if (currentPage < maxPage) {
            currentPage++;
            displayResults(currentResults);
        }
    });
      // Function to perform the search
    function performSearch(searchTerm) {
        // Show loading state
        searchResults.innerHTML = '<div class="loading">Searching for movies...</div>';
        
        // Check if noResults element exists before trying to hide it
        const noResultsElem = document.querySelector('.no-results');
        if (noResultsElem) {
            noResultsElem.style.display = 'none';
        }
        
        console.log("Performing search for:", searchTerm);
        console.log("Current page:", currentPage);
        console.log("Search input value:", searchInput.value);
        
        // Add a timestamp to prevent caching
        const timestamp = new Date().getTime();
        
        // Always use port 3000 for the API
        const apiPort = '3000';
        
        // Set base URL to localhost:3000
        const baseUrl = 'http://localhost:3000';
        console.log(`Using API base URL: ${baseUrl}`);
        
        // Check if the search term might be a genre rather than a movie title
        const commonGenres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                             'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                             'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'];
                             
        // More flexible genre detection - check if the input matches any genre case-insensitively
        // or if it's a partial match
        const searchTermLower = searchTerm.toLowerCase();
        const isGenreSearch = commonGenres.some(genre => 
            genre.toLowerCase() === searchTermLower || 
            searchTermLower.includes(genre.toLowerCase()) || 
            genre.toLowerCase().includes(searchTermLower));
            
        // Find the most closely matching genre if it is a genre search
        let matchedGenre = '';
        if (isGenreSearch) {
            matchedGenre = commonGenres.find(genre => 
                genre.toLowerCase() === searchTermLower || 
                searchTermLower.includes(genre.toLowerCase()) || 
                genre.toLowerCase().includes(searchTermLower)) || searchTerm;
                
            console.log(`Detected genre search for: ${matchedGenre}`);
        }
        
        // Use GET with URL parameters and cache-busting
        // If it's a genre search, include genre parameter to help the backend
        const searchUrl = isGenreSearch 
            ? `${baseUrl}/search?q=${encodeURIComponent(searchTerm)}&genre=${encodeURIComponent(matchedGenre)}&_=${timestamp}`
            : `${baseUrl}/search?q=${encodeURIComponent(searchTerm)}&title=${encodeURIComponent(searchTerm)}&_=${timestamp}`;
            
        console.log("Sending search request to:", searchUrl);
        
        fetch(searchUrl, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache'
            }
        })
        .then(response => {
            console.log("Search response status:", response.status);
            if (!response.ok) {
                searchResults.innerHTML = `<div class="error">Error searching for movies: Server responded with status: ${response.status}</div>`;
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Search response:", data);
            
            if (data.error) {
                console.warn("Search error:", data.error);
                searchResults.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                if (noResults) noResults.style.display = 'block';
                return;
            }
            
            const results = data.results;
            if (results && results.length > 0) {
                console.log(`Found ${results.length} results for: ${searchTerm}`);
                currentResults = results;
                totalResults = results.length;
                displayResults(results);
                
                // Check if noResults element exists before trying to hide it
                const noResultsElem = document.querySelector('.no-results');
                if (noResultsElem) {
                    noResultsElem.style.display = 'none';
                }
            } else {
                console.log("No results found for:", searchTerm);
                searchResults.innerHTML = `
                    <div class="no-results-message">
                        <h3>No results found</h3>
                        <p>We couldn't find any movies matching "${searchTerm}".</p>
                        <p>Try a different search term or check your spelling.</p>
                    </div>
                `;
                
                // Show no results section if it exists
                if (noResults) {
                    noResults.style.display = 'block';
                }
            }
            
            // Force the layout to be consistent
            enforceConsistentLayout();
        })        .catch(error => {
            console.error('Error during search:', error);
            searchResults.innerHTML = `
                <div class="error">
                    <h3>Error During Search</h3>
                    <p>There was a problem connecting to the movie database: ${error.message}</p>
                    <p class="error-info">Possible solutions:</p>
                    <ol class="error-steps">
                        <li>Check that the movie server is running on port 3000</li>
                        <li>Try refreshing the page</li>
                        <li>Check your network connection</li>
                    </ol>
                    <button onclick="location.reload()" class="retry-button">
                        Retry Search
                    </button>
                </div>
            `;
        });
    }
    
    // Function to display the results with pagination
    function displayResults(results) {
        // Add debugging logs
        console.log("displayResults called with " + results.length + " results");
        
        // Calculate pagination
        const start = (currentPage - 1) * resultsPerPage;
        const end = Math.min(start + resultsPerPage, results.length);
        const maxPage = Math.ceil(results.length / resultsPerPage);
        
        // Update pagination controls
        if (prevPageBtn) prevPageBtn.disabled = (currentPage <= 1);
        if (nextPageBtn) nextPageBtn.disabled = (currentPage >= maxPage);
        if (pageInfo) pageInfo.textContent = `Page ${currentPage} of ${maxPage}`;
        
        // Clear previous results
        searchResults.innerHTML = '';
        
        // Show current page of results
        const currentPageResults = results.slice(start, end);
        
        if (currentPageResults.length === 0) {
            searchResults.innerHTML = '<div class="no-results-message">No movies found matching your search.</div>';
            return;
        }
        
        // Create movie cards for each result
        currentPageResults.forEach((movie, index) => {
            console.log(`Creating card ${index + 1}/${currentPageResults.length} for movie:`, movie.title);
            try {
                const card = createMovieCard(movie);
                searchResults.appendChild(card);
            } catch (err) {
                console.error("Error creating movie card for", movie.title, ":", err);
                searchResults.innerHTML += `<div class="error">Error displaying movie: ${movie.title || 'Unknown'}</div>`;
            }
        });
        
        // Ensure layout stays consistent after results are loaded
        enforceConsistentLayout();
    }
      // Function to apply filters and sorting
    function applyFilters() {
        const sortValue = sortBySelect.value;
        
        // Create a copy of the current results to sort
        let sortedResults = [...currentResults];
        
        // Apply sorting
        switch(sortValue) {
            case 'rating-desc':
                sortedResults.sort((a, b) => {
                    // Handle null or undefined ratings
                    const ratingA = typeof a.rating === 'number' ? a.rating : parseFloat(a.rating || 0);
                    const ratingB = typeof b.rating === 'number' ? b.rating : parseFloat(b.rating || 0);
                    return ratingB - ratingA;
                });
                break;
            case 'rating-asc':
                sortedResults.sort((a, b) => {
                    // Handle null or undefined ratings
                    const ratingA = typeof a.rating === 'number' ? a.rating : parseFloat(a.rating || 0);
                    const ratingB = typeof b.rating === 'number' ? b.rating : parseFloat(b.rating || 0);
                    return ratingA - ratingB;
                });
                break;
            case 'year-desc':
                sortedResults.sort((a, b) => {
                    // Handle null or undefined years
                    const yearA = typeof a.year === 'number' ? a.year : parseFloat(a.year || 0);
                    const yearB = typeof b.year === 'number' ? b.year : parseFloat(b.year || 0);
                    return yearB - yearA;
                });
                break;
            case 'year-asc':
                sortedResults.sort((a, b) => {
                    // Handle null or undefined years
                    const yearA = typeof a.year === 'number' ? a.year : parseFloat(a.year || 0);
                    const yearB = typeof b.year === 'number' ? b.year : parseFloat(b.year || 0);
                    return yearA - yearB;
                });
                break;
        }
        
        // Reset to first page and display sorted results
        currentPage = 1;
        currentResults = sortedResults;
        displayResults(currentResults);
        
        // Show user feedback that filtering was applied
        const feedback = document.createElement('div');
        feedback.className = 'filter-feedback';
        feedback.textContent = 'Sorting applied';
        feedback.style.position = 'fixed';
        feedback.style.bottom = '20px';
        feedback.style.right = '20px';
        feedback.style.backgroundColor = 'rgba(0, 122, 255, 0.8)';
        feedback.style.color = 'white';
        feedback.style.padding = '10px 20px';
        feedback.style.borderRadius = '20px';
        feedback.style.zIndex = '9999';
        
        document.body.appendChild(feedback);
        
        // Remove feedback after 2 seconds
        setTimeout(() => {
            feedback.style.opacity = '0';
            feedback.style.transition = 'opacity 0.5s ease';
            setTimeout(() => document.body.removeChild(feedback), 500);
        }, 2000);
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
                    ratingDisplay = `<div class="movie-rating">Rating: ${movie.rating || 'N/A'}</div>`;
                }
            } else {
                ratingDisplay = `<div class="movie-rating">Rating: N/A</div>`;
            }
            
            const title = movie.title || 'Unknown Title';
            const year = movie.year || 'N/A';
            
            // Format genres for better display
            let genres = '';
            if (movie.genres) {
                if (typeof movie.genres === 'string') {
                    // Replace pipe delimiter with commas for better readability
                    genres = movie.genres.replace(/\|/g, ', ');
                } else if (movie.genre && typeof movie.genre === 'string') {
                    genres = movie.genre.replace(/\|/g, ', ');
                } else {
                    genres = 'N/A';
                }
            } else if (movie.genre) {
                if (typeof movie.genre === 'string') {
                    genres = movie.genre.replace(/\|/g, ', ');
                } else {
                    genres = 'N/A';
                }
            } else {
                genres = 'N/A';
            }
            
            card.innerHTML = `
                <a href="/search-results.html?q=${encodeURIComponent(title)}&recommend=true">
                    <div class="movie-info">
                        <div class="movie-title">${title}</div>
                        <div class="movie-year">${year}</div>
                        <div class="movie-genres">${genres}</div>
                        ${ratingDisplay}
                        <button class="recommend-similar" data-title="${title}">Similar Movies</button>
                    </div>
                </a>
            `;
            
            // Add event listener for the similar movies button
            setTimeout(() => {
                const recommendBtn = card.querySelector('.recommend-similar');
                if (recommendBtn) {
                    recommendBtn.addEventListener('click', function(e) {
                        e.preventDefault(); // Prevent the link from navigating
                        const title = this.getAttribute('data-title');
                        getRecommendations(title);
                    });
                }
            }, 0);
            
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
    }// Function to get recommendations for a movie
    function getRecommendations(movieTitle) {
        searchResults.innerHTML = '<div class="loading">Getting recommendations...</div>';
        
        // Check if the search term might be a genre rather than a movie title
        const commonGenres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'];
        
        // More flexible genre detection - check if the input matches any genre case-insensitively
        // or if it's a partial match (e.g., "scifi" matches "Sci-Fi")
        const movieTitleLower = movieTitle.toLowerCase();
        const isGenre = commonGenres.some(genre => 
            genre.toLowerCase() === movieTitleLower || 
            movieTitleLower.includes(genre.toLowerCase()) || 
            genre.toLowerCase().includes(movieTitleLower));
        
        // Update UI based on whether it's a genre or movie title
        if (isGenre) {
            // Find the closest matching genre for display purposes
            const matchedGenre = commonGenres.find(genre => 
                genre.toLowerCase() === movieTitleLower || 
                movieTitleLower.includes(genre.toLowerCase()) || 
                genre.toLowerCase().includes(movieTitleLower));
            
            searchQueryDisplay.textContent = `Movies in genre: ${matchedGenre || movieTitle}`;
            
            // For genres, we should do a regular search instead of recommendations
            performSearch(movieTitle);
            return;
        } else {
            searchQueryDisplay.textContent = `Recommendations for: ${movieTitle}`;
        }
        
        // Update URL to reflect recommendation mode
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('q', movieTitle);
        newUrl.searchParams.set('recommend', 'true');
        window.history.pushState({}, '', newUrl);
        
        currentQuery = movieTitle;
        
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        
        fetch(`http://localhost:3000/search?q=${encodeURIComponent(movieTitle)}&title=${encodeURIComponent(movieTitle)}&recommend=true&_=${timestamp}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache'
            }
        })
        .then(response => {
            if (!response.ok) {
                searchResults.innerHTML = `<div class="error">Error getting recommendations: Server responded with status: ${response.status}</div>`;
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Recommendation response:", data);
            
            if (data.error) {
                console.warn("Recommendation error:", data.error);
                searchResults.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                return;
            }
            
            const recommendations = data.recommendations;
            
            if (recommendations && recommendations.length > 0) {
                console.log(`Found ${recommendations.length} recommendations for: ${movieTitle}`);
                currentResults = recommendations;
                totalResults = recommendations.length;
                currentPage = 1;
                displayResults(recommendations);
            } else {
                console.log("No recommendations found for:", movieTitle);
                searchResults.innerHTML = `
                    <div class="no-results-message">
                        <h3>No recommendations found</h3>
                        <p>We couldn't find any recommendations for "${movieTitle}".</p>
                        <p>Try a different movie or check your spelling.</p>
                    </div>
                `;
            }
            
            // Force the layout to be consistent
            enforceConsistentLayout();
        })        .catch(error => {
            console.error('Error during recommendations:', error);
            searchResults.innerHTML = `
                <div class="error">
                    <h3>Error Getting Recommendations</h3>
                    <p>There was a problem connecting to the recommendation service: ${error.message}</p>
                    <p class="error-info">Possible solutions:</p>
                    <ol class="error-steps">
                        <li>Check that the movie server is running on port 3000</li>
                        <li>Try searching for a different movie</li>
                        <li>Try refreshing the page</li>
                    </ol>
                    <button onclick="location.reload()" class="retry-button">
                        Retry
                    </button>
                </div>
            `;
        });
    }
    
    // Function to enforce consistent layout in the UI
    function enforceConsistentLayout() {
        const heroSection = document.querySelector('.search-results-hero');
        if (heroSection) {
            heroSection.style.display = 'flex';
            heroSection.style.flexDirection = 'column';
            heroSection.style.alignItems = 'center';
            heroSection.style.justifyContent = 'center';
            heroSection.style.width = '100%';
            heroSection.style.padding = '60px 20px';
        }
    }
    
    // Check if API is available on port 3000
    function checkAPIAvailability() {
        console.log("Checking API availability on port 3000...");
        
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        
        // Try to access the API status endpoint
        fetch(`http://localhost:3000/api-status?_=${timestamp}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache'
            }
        })
        .then(response => {
            if (response.ok) {
                console.log("API is available on port 3000");
                return response.json();
            } else {
                throw new Error(`API returned status ${response.status}`);
            }
        })
        .then(data => {
            console.log("API Status:", data);
            
            // Add a custom event to indicate the API is ready
            const apiReadyEvent = new CustomEvent('apiReady', { detail: { port: 3000 } });
            document.dispatchEvent(apiReadyEvent);
            
            // Trigger search if there's a search term in URL
            const urlParams = new URLSearchParams(window.location.search);
            const searchTerm = urlParams.get('q');
            const isRecommend = urlParams.get('recommend') === 'true';
            
            if (searchTerm) {
                if (isRecommend) {
                    getRecommendations(searchTerm);
                } else {
                    performSearch(searchTerm);
                }
            }
        })
        .catch(error => {            console.error("API is not available on port 3000:", error);
            
            // If in search results and there's an error, show it
            const searchResults = document.getElementById('searchResults');
            if (searchResults) {
                searchResults.innerHTML = `
                    <div class="error">
                        <h3>Server Connection Error</h3>
                        <p>Could not connect to the movie recommendation server on port 3000. Please make sure the server is running.</p>
                        <p class="error-info">Steps to start the server:</p>
                        <ol class="error-steps">
                            <li>Open a terminal/command prompt in your movie recommendation system folder</li>
                            <li>Run the command: <code>python backend_df_direct_fixed.py</code></li>
                            <li>Wait for the message "Running on http://0.0.0.0:3000/"</li>
                            <li>Click the "Retry Connection" button below</li>
                        </ol>
                        <button id="retry-connection" class="retry-button">
                            Retry Connection
                        </button>
                    </div>
                `;
                
                // Force the hero section to stay consistent with server running layout
                enforceConsistentLayout();
                
                // Add event listener to the retry button
                const retryBtn = document.getElementById('retry-connection');
                if (retryBtn) {
                    retryBtn.addEventListener('click', function() {
                        searchResults.innerHTML = '<div class="loading">Retrying server connection...</div>';
                        setTimeout(checkAPIAvailability, 500);
                    });
                }
            }
        });
    }
});
