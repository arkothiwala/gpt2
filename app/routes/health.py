from fastapi import APIRouter

# HealthCheck API Router to monitor service health
health_check = APIRouter()


@health_check.get("/_health")
async def get_health_check():
    """
    GET command to check health status of API
    Registered as API Router to BaseFastAPI class
    Args:
        None
    Returns:
        JSON response {'health': 'service is healthy'} with Status 200 for healthy service
    """
    response = {"health": "service is healthy"}
    return response