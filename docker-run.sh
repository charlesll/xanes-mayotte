#!/bin/bash
# Script to build and run the Docker container
# Usage: ./docker-run.sh [build|run|exec|stop|clean]

set -e  # Stop on error

IMAGE_NAME="mayotte-xas"
IMAGE_TAG="1.0.0"
CONTAINER_NAME="mayotte_xas_analysis"

# Colors for display
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Help function
show_help() {
    echo -e "${BLUE}Usage:${NC} ./docker-run.sh [COMMAND]"
    echo ""
    echo "Available commands:"
    echo "  build   - Build Docker image"
    echo "  run     - Run XAS analysis in a new container"
    echo "  model   - Run thermodynamic modelling"
    echo "  all     - Run XAS + modelling (full pipeline)"
    echo "  exec    - Open shell in running container"
    echo "  stop    - Stop the container"
    echo "  clean   - Remove container and image"
    echo "  logs    - Show container logs"
    echo "  help    - Show this help"
    echo ""
    echo "Examples:"
    echo "  ./docker-run.sh build      # Build the image"
    echo "  ./docker-run.sh run        # Run XAS analysis"
    echo "  ./docker-run.sh model      # Run modelling"
    echo "  ./docker-run.sh all        # Full pipeline (XAS + modelling)"
    echo "  ./docker-run.sh exec       # Open shell"
}

# Function to build the image
build_image() {
    echo -e "${BLUE}=== Building Docker Image ===${NC}"
    
    # Check that Dockerfile exists
    if [ ! -f "Dockerfile" ]; then
        echo -e "${RED}Error: Dockerfile not found${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Building ${IMAGE_NAME}:${IMAGE_TAG}...${NC}"
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    
    # Also tag as 'latest'
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
    
    echo -e "${GREEN}✓ Image built successfully${NC}"
    echo ""
    docker images | grep ${IMAGE_NAME}
}

# Function to run analysis
run_analysis() {
    echo -e "${BLUE}=== Running XAS Analysis ===${NC}"
    
    # Check that image exists
    if ! docker images | grep -q ${IMAGE_NAME}; then
        echo -e "${YELLOW}Image not found. Building...${NC}"
        build_image
    fi
    
    # Create output directories if they don't exist
    mkdir -p figures/Iron figures/Sulfur results
    
    # Check input files
    if [ ! -f "tables/liste.xlsx" ]; then
        echo -e "${RED}⚠ Warning: liste.xlsx not found${NC}"
    fi
    if [ ! -d "xas/iron" ] || [ ! -d "xas/sulfur" ]; then
        echo -e "${RED}⚠ Warning: xas/iron or xas/sulfur directories missing${NC}"
    fi
    
    echo -e "${YELLOW}Starting container...${NC}"
    
    # Exécuter avec docker run
    docker run --rm \
        --name ${CONTAINER_NAME} \
        -v "$(pwd)/xas:/home/xasuser/mayotte/xas:ro" \
        -v "$(pwd)/tables/liste.xlsx:/home/xasuser/mayotte/tables/liste.xlsx:ro" \
        -v "$(pwd)/figures:/home/xasuser/mayotte/figures" \
        -v "$(pwd)/results:/home/xasuser/mayotte/results" \
        ${IMAGE_NAME}:${IMAGE_TAG}
    
    echo ""
    echo -e "${GREEN}✓ Analysis completed${NC}"
    echo -e "Results available in:"
    echo -e "  - ${BLUE}./figures/Iron/${NC}"
    echo -e "  - ${BLUE}./figures/Sulfur/${NC}"
    echo -e "  - ${BLUE}./results/${NC}"
}

# Function to run modelling
run_modelling() {
    echo -e "${BLUE}=== Thermodynamic Modelling (Moretti 2005) ===${NC}"
    
    # Check that image exists
    if ! docker images | grep -q ${IMAGE_NAME}; then
        echo -e "${YELLOW}Image not found. Building...${NC}"
        build_image
    fi
    
    # Create output directories if they don't exist
    mkdir -p figures/Modelling results/modelling
    
    # Check input files
    if [ ! -f "results/Results_synthese.xlsx" ]; then
        echo -e "${RED}⚠ Error: results/Results_synthese.xlsx not found${NC}"
        exit 1
    fi
    if [ ! -f "src/ctsfg6.for" ]; then
        echo -e "${RED}⚠ Error: src/ctsfg6.for not found${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Starting modelling...${NC}"
    
    # Exécuter avec docker run
    docker run --rm \
        --name ${CONTAINER_NAME}_model \
        -v "$(pwd)/figures:/home/xasuser/mayotte/figures" \
        -v "$(pwd)/results:/home/xasuser/mayotte/results" \
        ${IMAGE_NAME}:${IMAGE_TAG} python modelling.py
    
    echo ""
    echo -e "${GREEN}✓ Modelling completed${NC}"
    echo -e "Results available in:"
    echo -e "  - ${BLUE}./figures/Modelling/${NC}"
    echo -e "  - ${BLUE}./results/modelling/${NC}"
}

# Function to run full pipeline
run_all() {
    echo -e "${BLUE}=== Full Pipeline: XAS + Modelling ===${NC}"
    
    # Check that image exists
    if ! docker images | grep -q ${IMAGE_NAME}; then
        echo -e "${YELLOW}Image not found. Building...${NC}"
        build_image
    fi
    
    # Create all output directories
    mkdir -p figures/Iron figures/Sulfur figures/Modelling results/modelling
    
    # Check all input files
    if [ ! -f "tables/liste.xlsx" ]; then
        echo -e "${RED}⚠ Warning: tables/liste.xlsx not found${NC}"
    fi
    if [ ! -d "xas/iron" ] || [ ! -d "xas/sulfur" ]; then
        echo -e "${RED}⚠ Warning: xas/iron or xas/sulfur directories missing${NC}"
    fi
    if [ ! -f "results/Results_synthese.xlsx" ]; then
        echo -e "${RED}⚠ Warning: results/Results_synthese.xlsx not found${NC}"
    fi
    
    echo -e "${YELLOW}Starting full pipeline...${NC}"
    echo ""
    
    # Exécuter avec docker run
    docker run --rm \
        --name ${CONTAINER_NAME}_all \
        -v "$(pwd)/xas:/home/xasuser/mayotte/xas:ro" \
        -v "$(pwd)/tables/liste.xlsx:/home/xasuser/mayotte/tables/liste.xlsx:ro" \
        -v "$(pwd)/figures:/home/xasuser/mayotte/figures" \
        -v "$(pwd)/results:/home/xasuser/mayotte/results" \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        bash -c "python analysis_publication.py && echo '' && echo '════════════════════════════════════════' && echo 'Starting modelling...' && echo '════════════════════════════════════════' && echo '' && python modelling.py"
    
    echo ""
    echo -e "${GREEN}✓ Full pipeline completed${NC}"
    echo -e "XAS Results:"
    echo -e "  - ${BLUE}./figures/Iron/${NC}"
    echo -e "  - ${BLUE}./figures/Sulfur/${NC}"
    echo -e "Modelling Results:"
    echo -e "  - ${BLUE}./figures/Modelling/${NC}"
    echo -e "  - ${BLUE}./results/modelling/${NC}"
}

# Function to open shell
exec_shell() {
    echo -e "${BLUE}=== Interactive Shell ===${NC}"
    
    # Check if container is running
    if docker ps | grep -q ${CONTAINER_NAME}; then
        echo -e "${YELLOW}Connecting to running container...${NC}"
        docker exec -it ${CONTAINER_NAME} /bin/bash
    else
        # Start new container in interactive mode
        echo -e "${YELLOW}Starting new interactive container...${NC}"
        docker run --rm -it \
            --name ${CONTAINER_NAME}_shell \
            -v "$(pwd)/xas:/home/xasuser/mayotte/xas:ro" \
            -v "$(pwd)/tables/liste.xlsx:/home/xasuser/mayotte/tables/liste.xlsx:ro" \
            -v "$(pwd)/figures:/home/xasuser/mayotte/figures" \
            -v "$(pwd)/results:/home/xasuser/mayotte/results" \
            --entrypoint /bin/bash \
            ${IMAGE_NAME}:${IMAGE_TAG}
    fi
}

# Function to stop container
stop_container() {
    echo -e "${BLUE}=== Stopping Container ===${NC}"
    
    if docker ps | grep -q ${CONTAINER_NAME}; then
        echo -e "${YELLOW}Stopping ${CONTAINER_NAME}...${NC}"
        docker stop ${CONTAINER_NAME}
        echo -e "${GREEN}✓ Container stopped${NC}"
    else
        echo -e "${YELLOW}No container running${NC}"
    fi
}

# Function to clean up
clean_all() {
    echo -e "${BLUE}=== Cleanup ===${NC}"
    
    # Stop container if running
    if docker ps | grep -q ${CONTAINER_NAME}; then
        echo -e "${YELLOW}Stopping container...${NC}"
        docker stop ${CONTAINER_NAME}
    fi
    
    # Remove container if exists
    if docker ps -a | grep -q ${CONTAINER_NAME}; then
        echo -e "${YELLOW}Removing container...${NC}"
        docker rm ${CONTAINER_NAME}
    fi
    
    # Remove image
    if docker images | grep -q ${IMAGE_NAME}; then
        echo -e "${YELLOW}Removing image...${NC}"
        docker rmi ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✓ Cleanup completed${NC}"
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}=== Container Logs ===${NC}"
    
    if docker ps | grep -q ${CONTAINER_NAME}; then
        docker logs -f ${CONTAINER_NAME}
    else
        echo -e "${YELLOW}No container running${NC}"
    fi
}

# Function to use docker-compose
compose_up() {
    echo -e "${BLUE}=== Starting with Docker Compose ===${NC}"
    
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}Error: docker-compose.yml not found${NC}"
        exit 1
    fi
    
    docker-compose up --build
}

# Main menu
case "${1:-help}" in
    build)
        build_image
        ;;
    run)
        run_analysis
        ;;
    model|modelling)
        run_modelling
        ;;
    all|full|pipeline)
        run_all
        ;;
    exec|shell)
        exec_shell
        ;;
    stop)
        stop_container
        ;;
    clean)
        clean_all
        ;;
    logs)
        show_logs
        ;;
    compose)
        compose_up
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
