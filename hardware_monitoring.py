from fastapi import APIRouter
from fastapi.responses import JSONResponse
import requests
from app.config import PROMETHEUS_HOST, CPU_INSTANCE, GPU_INSTANCE, GPU_ID
from app.logging_config import logger

router = APIRouter(prefix="/hardware", tags=["hardware"])

@router.get('/')
def main():
    return {'msg':'hardware server is running'}
@router.get("/metrics")
async def get_metrics():
    queries = {
        'CPU_usage': f'(1 - avg by(instance)(rate(node_cpu_seconds_total{{instance="{CPU_INSTANCE}", mode="idle"}}[1m]))) * 100',
        'RAM_usage': f'(1 - node_memory_MemAvailable_bytes{{instance="{CPU_INSTANCE}"}} / node_memory_MemTotal_bytes{{instance="{CPU_INSTANCE}"}}) * 100',
        'GPU_usage': f'DCGM_FI_DEV_GPU_UTIL{{instance="{GPU_INSTANCE}", gpu="{GPU_ID}"}}',
        'GPU_ram_usage': f'DCGM_FI_DEV_FB_USED{{instance="{GPU_INSTANCE}", gpu="{GPU_ID}"}}'
    }
    results = {}
    try:
        logger.info("Starting metrics collection")
        for key, query in queries.items():
            url = f'{PROMETHEUS_HOST}/api/v1/query'
            params = {'query': query}
            logger.info(f"Querying {key}: {query}")

            response = requests.get(url, params=params)
            logger.info(f"Response status code for {key}: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Response data for {key}: {data}")

                if data['status'] == 'success' and data['data']['result']:
                    val = float(data['data']['result'][0]['value'][1]).__round__(2)
                    results[key] = val
                    logger.info(f"{key} value: {val}")
                else:
                    results[key] = None
                    logger.warning(f"{key} has no result data.")
            else:
                results[key] = None
                logger.error(f"Failed to fetch {key}: HTTP status {response.status_code}")

        logger.info("Metrics collection completed")
        return JSONResponse(content=results)

    except Exception as e:
        logger.exception("An error occurred while fetching metrics")
        return JSONResponse(content={'error': str(e)})
