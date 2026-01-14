import pyvista as pv
from pyvistaqt import BackgroundPlotter
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_pyvistaqt():
    logging.info("Initializing PyVistaQt BackgroundPlotter...")
    try:
        plotter = BackgroundPlotter(title="PyVistaQt Test", auto_update=True)
        plotter.add_axes()

        # Add a simple mesh
        mesh = pv.Sphere()
        plotter.add_mesh(mesh, color='red', show_edges=True)

        logging.info("PyVistaQt window should be visible. Close it to exit.")
        # Keep the plotter window open until closed by user
        plotter.app.exec_() 
        # For some systems, plotter.app.exec_() might block indefinitely or
        # require manual closing. If it doesn't return, user might need to Ctrl+C.
        
    except Exception as e:
        logging.error(f"Error during PyVistaQt test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_pyvistaqt()
