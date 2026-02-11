import uvicorn
import os
import sys

# Ensure backend in path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from api import app
    print("Imported app successfully.")
    # Run on 8001 to avoid conflict
    uvicorn.run(app, host="0.0.0.0", port=8001)
except Exception as e:
    print(f"Failed to start server: {e}")
    import traceback
    traceback.print_exc()
