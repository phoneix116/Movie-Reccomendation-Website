// Search Results Page Script
document.addEventListener('DOMContentLoaded', function() {    // Page elements
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
    
    // Check if server is available on port 3000 directly
    checkAPIAvailability();
    
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
    
    // Function to ensure consistent layout across all states
    function ensureConsistentLayout() {
        // Force the hero section to maintain a consistent appearance
        const heroSection = document.querySelector('.search-results-hero');
        if (heroSection) {
            heroSection.style.display = 'flex';
            heroSection.style.flexDirection = 'column';
            heroSection.style.alignItems = 'center';
            heroSection.style.justifyContent = 'center';
            heroSection.style.width = '100%';
            heroSection.style.padding = '60px 20px';
        }
        
        // Ensure content header stays consistent
        const contentHeader = document.querySelector('.content-header');
        if (contentHeader) {
            contentHeader.style.display = 'flex';
            contentHeader.style.justifyContent = 'space-between';
            contentHeader.style.alignItems = 'center';
            contentHeader.style.flexWrap = 'wrap';
        }
    }
    
    // Call this function on page load and after any major DOM updates
    ensureConsistentLayout();
    
    // Also add a window resize listener to maintain consistency
    window.addEventListener('resize', ensureConsistentLayout);
    
    // Pagination handlers
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
    });    // Function to perform the search    function performSearch(searchTerm) {
        // Show loading state
        searchResults.innerHTML = '<div class="loading">Searching for movies...</div>';
        
        // Check if noResults element exists before trying to hide it
        const noResultsElem = document.querySelector('.no-results');
        if (noResultsElem) {
            noResultsElem.style.display = 'none';
        }
        
        console.log("Performing search for:", searchTerm);        // Add a timestamp to prevent caching
        const timestamp = new Date().getTime();
        
        // Always use port 3000 for the API
        const apiPort = '3000';
        
        // Set base URL to localhost:3000
        const baseUrl = 'http://localhost:3000';
        console.log(`Using API base URL: ${baseUrl}`);
        
        // Use GET with URL parameters and cache-busting
        fetch(`${baseUrl}/search?q=${encodeURIComponent(searchTerm)}&title=${encodeURIComponent(searchTerm)}&_=${timestamp}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache'
            }
        })
        .then(response => {
            if (!response.ok) {
                searchResults.innerHTML = `<div class="error">Error searching for movies: Server responded with status: ${response.status}</div>`;
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        }).then(data => {
            console.log("Search response:", data);
            
            if (data.error) {
                console.warn("Search error:", data.error);
                searchResults.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                noResults.style.display = 'block';
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
                searchResults.innerHTML = '';
                
                // Check if noResults element exists before trying to show it
                const noResultsElem = document.querySelector('.no-results');
                if (noResultsElem) {
                    noResultsElem.style.display = 'block';
                } else {
                    // If no-results element doesn't exist, create a message in the search results area
                    searchResults.innerHTML = '<div class="no-results-message">No movies found matching your search.</div>';
                }
            }
        })
        .catch(error => {
            console.error('Error during search:', error);
            searchResults.innerHTML = `<div class="error">Error searching for movies: ${error.message}</div>`;
        });
    }
    
    // Function to display the results with pagination    function displayResults(results) {
        // Add debugging logs
        console.log("displayResults called with", results.length, "results");
        
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
                sortedResults.sort((a, b) => b.rating - a.rating);
                break;
            case 'rating-asc':
                sortedResults.sort((a, b) => a.rating - b.rating);
                break;
            case 'year-desc':
                sortedResults.sort((a, b) => b.year - a.year);
                break;
            case 'year-asc':
                sortedResults.sort((a, b) => a.year - b.year);
                break;
        }
        
        // Reset to first page and display sorted results
        currentPage = 1;
        currentResults = sortedResults;
        displayResults(currentResults);
    }
    
    // Function to create a movie card    function createMovieCard(movie) {
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
            const genres = movie.genres || movie.genre || 'N/A';              card.innerHTML = `
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
                    recommendBtn.addEventListener('click', function() {
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
    }      // Function to get recommendations for a movie
    function getRecommendations(movieTitle) {
        searchResults.innerHTML = '<div class="loading">Getting recommendations...</div>';
        searchQueryDisplay.textContent = `Recommendations for: ${movieTitle}`;
        
        // Update URL to reflect recommendation mode
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('q', movieTitle);
        newUrl.searchParams.set('recommend', 'true');
        window.history.pushState({}, '', newUrl);
        
        console.log("Getting recommendations for:", movieTitle);        // Add a timestamp to prevent caching
        const timestamp = new Date().getTime();
        
        // Always use port 3000 for the API
        const apiPort = '3000';
        
        // Set base URL to localhost:3000
        const baseUrl = 'http://localhost:3000';
        console.log(`Using API base URL: ${baseUrl} for recommendations`);
        
        // Try to use the search endpoint with recommend=true parameter
        fetch(`${baseUrl}/search?title=${encodeURIComponent(movieTitle)}&recommend=true&_=${timestamp}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache'
            }
        }).then(response => {
            if (!response.ok) {
                searchResults.innerHTML = `<div class="error">Error getting recommendations: Server responded with status: ${response.status}</div>`;
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })        .then(data => {
            if (data.error) {
                searchResults.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                return;
            }
              // Support both API formats - recommendations from /recommend or results from /search
            const results = data.recommendations || data.results || [];
            if (results && results.length > 0) {
                console.log(`Found ${results.length} recommendations for: ${movieTitle}`);
                currentResults = results;
                totalResults = results.length;
                currentPage = 1;
                displayResults(results);
                
                // Check if noResults element exists before trying to hide it
                const noResultsElem = document.querySelector('.no-results');
                if (noResultsElem) {
                    noResultsElem.style.display = 'none';
                }
            } else {
                searchResults.innerHTML = '<div class="no-results-message">No recommendations found.</div>';
                
                // Check if noResults element exists before trying to show it
                const noResultsElem = document.querySelector('.no-results');
                if (noResultsElem) {
                    noResultsElem.style.display = 'block';
                }
            }
        })
        .catch(error => {
            console.error('Error getting recommendations:', error);
            searchResults.innerHTML = `<div class="error">Error getting recommendations: ${error.message}</div>`;
        });
    }
    
    // If there's a recommendation button in the no results area
    const noResultsRecommendBtn = document.querySelector('.no-results .recommendation-btn');
    if (noResultsRecommendBtn) {
        noResultsRecommendBtn.addEventListener('click', function() {
            const popularMovies = ["Toy Story", "Jumanji", "Heat", "Pulp Fiction", "The Shawshank Redemption"];
            const randomMovie = popularMovies[Math.floor(Math.random() * popularMovies.length)];
            getRecommendations(randomMovie);
        });
    }
    
    // Add event listeners for select focus/blur to enhance styling
    if (sortBySelect) {
        sortBySelect.addEventListener('focus', function() {
            this.classList.add('active');
        });
        
        sortBySelect.addEventListener('blur', function() {
            this.classList.remove('active');
        });
    }
    
    // Function to ensure consistent UI regardless of server state
function enforceConsistentLayout() {
    // Ensure search hero section has consistent styling
    const heroSection = document.querySelector('.search-results-hero');
    if (heroSection) {
        heroSection.style.display = 'flex';
        heroSection.style.flexDirection = 'column';
        heroSection.style.alignItems = 'center';
        heroSection.style.justifyContent = 'center';
        heroSection.style.padding = '40px 20px';
    }
    
    // Ensure content header has proper spacing
    const contentHeader = document.querySelector('.content-header');
    if (contentHeader) {
        contentHeader.style.display = 'flex';
        contentHeader.style.justifyContent = 'space-between';
        contentHeader.style.flexWrap = 'wrap';
        contentHeader.style.gap = '15px';
    }
}

// Function to check API availability on port 3000
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
    .then(response => {        if (response.ok) {
            console.log("API is available on port 3000");
            return response.json();
        } else {
            throw new Error(`API returned status ${response.status}`);
        }
    })
    .then(data => {
        console.log("API Status:", data);
        
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
    .catch(error => {        console.error("API is not available on port 3000:", error);
        
        // If in search results and there's an error, show it
        const searchResults = document.getElementById('searchResults');
        if (searchResults) {
            searchResults.innerHTML = `
                <div class="error">
                    <h3>Server Connection Error</h3>
                    <p>Could not connect to the movie recommendation server on port 3000. Please make sure the server is running.</p>
                    <p class="error-info">Steps to start the server:</p>
                    <ol class="error-steps">
                        <li>Find and run the file "run_on_3000.bat" in your movie recommendation system folder</li>
                        <li>Wait for the server to start</li>
                        <li>Click the "Retry Connection" button below</li>
                    </ol>
                    <button id="retry-api" class="retry-button">
                        Retry Connection
                    </button>
                </div>
            `;
            
            // Add retry button functionality
            const retryBtn = document.getElementById('retry-api');
            if (retryBtn) {
                retryBtn.addEventListener('click', function() {
                    searchResults.innerHTML = '<div class="loading">Retrying server connection...</div>';
                    setTimeout(checkAPIAvailability, 500);
                });
            }
        }
    });
}

// Call this function at DOMContentLoaded and when search results are updated
document.addEventListener('DOMContentLoaded', enforceConsistentLayout);

// Additional function to be called when search results are loaded
function onSearchResultsLoaded() {
    enforceConsistentLayout();
}
});
