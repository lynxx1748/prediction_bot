"""
Asynchronous utilities for the trading bot.
"""

import asyncio
import traceback
import concurrent.futures
import time
import logging

logger = logging.getLogger(__name__)

async def timeout_resistant_prediction(prediction_coroutine, timeout=2.0, default_result=("BULL", 0.51)):
    """
    Execute a prediction coroutine with better timeout handling
    
    Args:
        prediction_coroutine: The async coroutine to execute
        timeout: Maximum time to wait (seconds)
        default_result: Default result to return on timeout
        
    Returns:
        The prediction result or default on timeout
    """
    try:
        # Create a task from the coroutine
        task = asyncio.create_task(prediction_coroutine)
        
        # Wait for the task with a timeout
        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Try to cancel the task if it's still running
            if not task.done():
                task.cancel()
                
            logger.warning(f"⚠️ Async prediction timed out after {timeout} seconds - using fallback")
            return default_result
            
    except Exception as e:
        logger.error(f"❌ Error in async prediction handler: {e}")
        traceback.print_exc()
        return default_result

def run_prediction_with_timeout(async_function, data, timeout=2.0, default_result=("BULL", 0.51)):
    """
    Run an async prediction function with proper timeout handling
    
    Args:
        async_function: The async function to run
        data: Data to pass to the function
        timeout: Maximum time to wait (seconds)
        default_result: Default result on timeout
        
    Returns:
        The prediction result or default on timeout
    """
    try:
        loop = asyncio.get_event_loop()
        
        # Create the coroutine
        coroutine = async_function(data)
        
        # If the loop is already running, use the thread-safe approach
        if loop.is_running():
            # Create a Future for the result
            future = asyncio.run_coroutine_threadsafe(timeout_resistant_prediction(coroutine, timeout), loop)
            
            try:
                # Wait for the result with a slightly longer timeout
                return future.result(timeout=timeout + 0.5)
            except concurrent.futures.TimeoutError:
                logger.warning(f"⚠️ Thread-safe async execution timed out after {timeout + 0.5} seconds")
                return default_result
        else:
            # Normal case - run the coroutine with timeout handling
            return loop.run_until_complete(timeout_resistant_prediction(coroutine, timeout))
            
    except Exception as e:
        logger.error(f"❌ Error running prediction with timeout: {e}")
        traceback.print_exc()
        return default_result

async def gather_with_concurrency(n, *tasks):
    """
    Run coroutines with a concurrency limit.
    
    Args:
        n: Maximum number of concurrent tasks
        *tasks: Tasks to run
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*(sem_task(task) for task in tasks)) 