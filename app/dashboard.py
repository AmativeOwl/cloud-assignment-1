import io
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for server-side rendering
import matplotlib.pyplot as plt

from app import carpark_cache, request_tracker


def generate_dashboard_png() -> bytes:
    known = carpark_cache.get_all_known()

    carpark_ids = sorted(known.keys())
    available = [known[cid]["available_spaces"] for cid in carpark_ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: available spaces per car park
    if carpark_ids:
        axes[0].bar(carpark_ids, available, color="steelblue")
        axes[0].set_title("Available Spaces per Car Park")
        axes[0].set_xlabel("Car Park ID")
        axes[0].set_ylabel("Available Spaces")
        axes[0].tick_params(axis="x", rotation=90)
    else:
        axes[0].text(0.5, 0.5, "No data yet", ha="center", va="center")
        axes[0].set_title("Available Spaces per Car Park")

    # Right: recent activity summary
    recent_requests = request_tracker.count_recent_requests()
    recent_users = request_tracker.count_recent_unique_users()
    axes[1].bar(
        ["Requests (30s)", "Unique Users (30s)"],
        [recent_requests, recent_users],
        color=["darkorange", "seagreen"],
    )
    axes[1].set_title("Recent Platform Activity")
    axes[1].set_ylabel("Count")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()