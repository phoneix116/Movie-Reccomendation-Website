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
    const baseUrl = window.location.origin;
    
    // Add recommendation button functionality
    const recommendationBtn = document.querySelector('.recommendation-btn');
    const recommendationModal = document.getElementById('recommendationModal');
    const closeModalBtn = document.querySelector('.close-modal');
    const recommendationInput = document.getElementById('recommendationInput');
    const recommendationSearchButton = document.getElementById('recommendationSearchButton');
    const popularMovies = document.querySelectorAll('.popular-movie');
    
    // Handle recommendation button click
    if (recommendationBtn && recommendationModal) {
        recommendationBtn.addEventListener('click', function(e) {
            e.preventDefault();
            recommendationModal.style.display = 'block';
            if (recommendationInput) recommendationInput.focus();
        });
    }
    
    // Close modal when clicking the close button
    if (closeModalBtn && recommendationModal) {
        closeModalBtn.addEventListener('click', function() {
            recommendationModal.style.display = 'none';
        });
    }
    
    // Close modal when clicking outside of it
    window.addEventListener('click', function(event) {
        if (event.target === recommendationModal) {
            recommendationModal.style.display = 'none';
        }
    });
    
    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && recommendationModal && recommendationModal.style.display === 'block') {
            recommendationModal.style.display = 'none';
        }
    });
    
    // Handle recommendation search button
    if (recommendationSearchButton) {
        recommendationSearchButton.addEventListener('click', function() {
            if (recommendationInput && recommendationInput.value.trim() !== '') {
                const searchTerm = encodeURIComponent(recommendationInput.value.trim());
                window.location.href = `/search-results.html?q=${searchTerm}&recommend=true`;
            }
        });
    }
    
    // Handle recommendation input enter key
    if (recommendationInput) {
        recommendationInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && this.value.trim() !== '') {
                const searchTerm = encodeURIComponent(this.value.trim());
                window.location.href = `/search-results.html?q=${searchTerm}&recommend=true`;
            }
        });
    }
    
    // Handle popular movie clicks
    popularMovies.forEach(movie => {
        movie.addEventListener('click', function() {
            const movieTitle = this.getAttribute('data-title');
            if (movieTitle) {
                window.location.href = `/search-results.html?q=${encodeURIComponent(movieTitle)}&recommend=true`;
            }
        });
    });
    
    // Handle discover search button
    const discoverSearchButton = document.getElementById('discoverSearchButton');
    const discoverSearchInput = document.getElementById('discoverSearchInput');
    
    if (discoverSearchButton) {
        discoverSearchButton.addEventListener('click', function() {
            if (discoverSearchInput && discoverSearchInput.value.trim() !== '') {
                const searchTerm = encodeURIComponent(discoverSearchInput.value.trim());
                window.location.href = `/search-results.html?q=${searchTerm}`;
            }
        });
    }
    
    // Handle discover search input enter key
    if (discoverSearchInput) {
        discoverSearchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && this.value.trim() !== '') {
                const searchTerm = encodeURIComponent(this.value.trim());
                window.location.href = `/search-results.html?q=${searchTerm}`;
            }
        });
    }

    // Display sample movies in each category
    displaySampleMovies();

    // Function to display sample movies
    function displaySampleMovies() {
        // Sample movies for each genre
        const genres = {
            'action': [
                { title: 'The Dark Knight', year: '2008', genres: 'Action/Crime/Drama', rating: 9.0 },
                { title: 'Inception', year: '2010', genres: 'Action/Adventure/Sci-Fi', rating: 8.8 },
                { title: 'Die Hard', year: '1988', genres: 'Action/Thriller', rating: 8.2 },
                { title: 'Mad Max: Fury Road', year: '2015', genres: 'Action/Adventure/Sci-Fi', rating: 8.1 },
                { title: 'John Wick', year: '2014', genres: 'Action/Crime/Thriller', rating: 7.4 },
                { title: 'The Matrix', year: '1999', genres: 'Action/Sci-Fi', rating: 8.7 }
            ],
            'comedy': [
                { title: 'The Grand Budapest Hotel', year: '2014', genres: 'Adventure/Comedy/Crime', rating: 8.1 },
                { title: 'Superbad', year: '2007', genres: 'Comedy', rating: 7.6 },
                { title: 'The Hangover', year: '2009', genres: 'Comedy', rating: 7.7 },
                { title: 'Bridesmaids', year: '2011', genres: 'Comedy/Romance', rating: 6.8 },
                { title: 'Anchorman', year: '2004', genres: 'Comedy', rating: 7.2 },
                { title: 'Step Brothers', year: '2008', genres: 'Comedy', rating: 6.9 }
            ],
            'drama': [
                { title: 'The Shawshank Redemption', year: '1994', genres: 'Drama', rating: 9.3 },
                { title: 'The Godfather', year: '1972', genres: 'Crime/Drama', rating: 9.2 },
                { title: 'Schindler\'s List', year: '1993', genres: 'Biography/Drama/History', rating: 8.9 },
                { title: 'Forrest Gump', year: '1994', genres: 'Drama/Romance', rating: 8.8 },
                { title: 'The Green Mile', year: '1999', genres: 'Crime/Drama/Fantasy', rating: 8.6 },
                { title: 'Parasite', year: '2019', genres: 'Comedy/Drama/Thriller', rating: 8.6 }
            ],
            'scifi': [
                { title: 'Blade Runner 2049', year: '2017', genres: 'Action/Drama/Mystery/Sci-Fi', rating: 8.0 },
                { title: 'Interstellar', year: '2014', genres: 'Adventure/Drama/Sci-Fi', rating: 8.6 },
                { title: 'Arrival', year: '2016', genres: 'Drama/Sci-Fi', rating: 7.9 },
                { title: 'The Martian', year: '2015', genres: 'Adventure/Drama/Sci-Fi', rating: 8.0 },
                { title: 'Dune', year: '2021', genres: 'Action/Adventure/Drama/Sci-Fi', rating: 8.0 },
                { title: 'Ex Machina', year: '2014', genres: 'Drama/Sci-Fi/Thriller', rating: 7.7 }
            ],
            'romance': [
                { title: 'The Notebook', year: '2004', genres: 'Drama/Romance', rating: 7.8 },
                { title: 'Eternal Sunshine of the Spotless Mind', year: '2004', genres: 'Drama/Romance/Sci-Fi', rating: 8.3 },
                { title: 'Pride & Prejudice', year: '2005', genres: 'Drama/Romance', rating: 7.8 },
                { title: 'La La Land', year: '2016', genres: 'Comedy/Drama/Music/Romance', rating: 8.0 },
                { title: 'Before Sunrise', year: '1995', genres: 'Drama/Romance', rating: 8.1 },
                { title: 'Titanic', year: '1997', genres: 'Drama/Romance', rating: 7.8 }
            ],
            'animation': [
                { title: 'Spirited Away', year: '2001', genres: 'Animation/Adventure/Family/Fantasy/Mystery', rating: 8.6 },
                { title: 'Your Name', year: '2016', genres: 'Animation/Drama/Fantasy/Romance', rating: 8.4 },
                { title: 'Toy Story', year: '1995', genres: 'Animation/Adventure/Comedy/Family/Fantasy', rating: 8.3 },
                { title: 'The Lion King', year: '1994', genres: 'Animation/Adventure/Drama/Family/Musical', rating: 8.5 },
                { title: 'Spider-Man: Into the Spider-Verse', year: '2018', genres: 'Animation/Action/Adventure', rating: 8.4 },
                { title: 'Coco', year: '2017', genres: 'Animation/Adventure/Drama/Family/Fantasy/Music/Mystery', rating: 8.4 }
            ]
        };

        // Display movies for each genre
        displayGenreMovies('action', genres.action, actionMoviesContainer);
        displayGenreMovies('comedy', genres.comedy, comedyMoviesContainer);
        displayGenreMovies('drama', genres.drama, dramaMoviesContainer);
        displayGenreMovies('scifi', genres.scifi, scifiMoviesContainer);
        displayGenreMovies('romance', genres.romance, romanceMoviesContainer);
        displayGenreMovies('animation', genres.animation, animationMoviesContainer);
    }

    // Function to display movies for a specific genre
    function displayGenreMovies(genreName, movies, container) {
        if (!container) return;

        // Clear container
        container.innerHTML = '';

        // Add movie cards
        movies.forEach(movie => {
            const card = document.createElement('div');
            card.className = 'movie-card';
            
            // Create link to search results with recommendation
            const movieUrl = `/search-results.html?q=${encodeURIComponent(movie.title)}&recommend=true`;
            
            // Create card content
            card.innerHTML = `
                <a href="${movieUrl}">
                    <div class="movie-info">
                        <div class="movie-title">${movie.title}</div>
                        <div class="movie-year">${movie.year}</div>
                        <div class="movie-genres">${movie.genres}</div>
                        <div class="movie-rating">Rating: ${movie.rating.toFixed(1)}</div>
                    </div>
                </a>
            `;
            
            container.appendChild(card);
        });
    }
});
