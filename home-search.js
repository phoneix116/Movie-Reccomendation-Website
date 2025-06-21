// Home page search functionality
document.addEventListener('DOMContentLoaded', function() {
    // Get the search input and button elements
    const searchInput = document.getElementById('homeSearchInput');
    const searchButton = document.getElementById('homeSearchButton');
    
    // Function to handle search submission
    function handleSearch() {
        const searchTerm = searchInput.value.trim();
        if (searchTerm) {
            // Add a timestamp to prevent caching
            const timestamp = new Date().getTime();
            
            // Redirect to search-results.html with the search query and cache-busting
            window.location.href = `search-results.html?q=${encodeURIComponent(searchTerm)}&_=${timestamp}`;
        }
    }
    
    // Add click event listener to the search button
    searchButton.addEventListener('click', handleSearch);
    
    // Also handle the "Enter" key in the search input
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });      // Also handle "Get Personalized Recommendations" button 
    const recommendationBtn = document.querySelector('.recommendation-btn');
    const recommendationModal = document.getElementById('recommendationModal');
    const closeModalBtn = document.querySelector('.close-modal');
    const recommendationInput = document.getElementById('recommendationInput');
    const recommendationSearchButton = document.getElementById('recommendationSearchButton');
    const popularMovieButtons = document.querySelectorAll('.popular-movie');
    
    if (recommendationBtn && recommendationModal) {
        // Function to open modal
        recommendationBtn.addEventListener('click', function(e) {
            e.preventDefault();
            recommendationModal.style.display = 'block';
            recommendationInput.focus();
        });
        
        // Function to close modal
        closeModalBtn.addEventListener('click', function() {
            recommendationModal.style.display = 'none';
        });
          // Close modal when clicking outside of it
        window.addEventListener('click', function(e) {
            if (e.target === recommendationModal) {
                recommendationModal.style.display = 'none';
            }
        });
        
        // Close modal with Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && recommendationModal.style.display === 'block') {
                recommendationModal.style.display = 'none';
            }
        });
          // Handle recommendation search button click
        if (recommendationSearchButton) {
            recommendationSearchButton.addEventListener('click', function() {
                const movieTitle = recommendationInput.value.trim();
                if (movieTitle) {
                    // Show loading state
                    this.classList.add('loading');
                    this.textContent = 'Finding...';
                    
                    // Add a short delay to show the loading state
                    setTimeout(() => {
                        goToRecommendations(movieTitle);
                    }, 300);
                }
            });
        }
        
        // Handle "Enter" key in recommendation input
        if (recommendationInput) {
            recommendationInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    const movieTitle = recommendationInput.value.trim();
                    if (movieTitle) {
                        // Show loading state on the button
                        if (recommendationSearchButton) {
                            recommendationSearchButton.classList.add('loading');
                            recommendationSearchButton.textContent = 'Finding...';
                        }
                        
                        // Add a short delay to show the loading state
                        setTimeout(() => {
                            goToRecommendations(movieTitle);
                        }, 300);
                    }
                }
            });
        }
          // Handle popular movie buttons
        if (popularMovieButtons.length > 0) {
            popularMovieButtons.forEach(button => {
                button.addEventListener('click', function() {
                    const movieTitle = this.dataset.title;
                    
                    // Add visual feedback with class
                    this.classList.add('popular-movie-clicked');
                    
                    // Small delay before navigation to show the animation
                    setTimeout(() => {
                        goToRecommendations(movieTitle);
                    }, 200);
                });
            });
        }
    }
    
    // Function to redirect to recommendations page
    function goToRecommendations(movieTitle) {
        // Add a timestamp to prevent caching
        const timestamp = new Date().getTime();
        window.location.href = `search-results.html?q=${encodeURIComponent(movieTitle)}&recommend=true&_=${timestamp}`;
    }// Preload the search API to warm up the server and determine port
    try {
        // Try multiple ports, starting with the most likely one
        const ports = [3000, 5000, 8000, 7000];
        let preloadSuccessful = false;
        
        function checkNextPort(index) {
            if (index >= ports.length) {
                console.warn("No API server found on any port");
                return;
            }
            
            const port = ports[index];
            // Add timestamp for cache busting
            const timestamp = new Date().getTime();
            
            fetch(`http://localhost:${port}/api-status?_=${timestamp}`, {
                headers: {
                    'Cache-Control': 'no-cache'
                }
            })
            .then(response => {
                if (response.ok) {
                    console.log(`API responding on port ${port}`);
                    localStorage.setItem('movieApiPort', port);
                    preloadSuccessful = true;
                    return response.json();
                } else {
                    throw new Error(`Server responded with ${response.status}`);
                }
            })
            .then(data => console.log('API Status:', data.status))
            .catch(err => {
                console.warn(`API preload error on port ${port}:`, err);
                // Try next port
                checkNextPort(index + 1);
            });
        }
        
        // Start checking first port
        checkNextPort(0);
    } catch (e) {
        console.warn('API preload failed:', e);
    }
});
