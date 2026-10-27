"""Men's international rugby union match predictor."""
import os

# The datasets are small (a few thousand rows): OpenMP threading in scikit-learn's gradient
# boosting costs more in overhead than it saves, so default to one thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
