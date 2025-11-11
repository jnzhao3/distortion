from matplotlib import pyplot as plt
import numpy as np

def ranking_text(ranking):
    """
    Convert a ranking list into a human-readable text format.

    Args:
        ranking (list): A list of items in ranked order.
    Returns:
        str: A formatted string representing the ranking.
    """
    # sort the ranking to ensure correct order
    ranked_items = sorted(ranking.keys(), key=lambda x: ranking.get(x), reverse=True)
    text_output = ""
    for i in ranked_items:
        text_output += f"Candidate {i}: Score {ranking[i]}\n"

    return text_output

def plot_ranking(ranking, title="Scores (High → Low)"):
    """
    Plot a bar chart of items and their scores, sorted high→low, 
    with colors ranging from green (high) to red (low).
    
    Args:
        items (list[str]): Names or labels for each item.
        scores (list[float]): Numeric scores.
        title (str): Plot title.
    """
    items = np.array(list(ranking.keys()))
    scores = np.array(list(ranking.values()))
    
    # Sort by score descending
    sort_idx = np.argsort(-scores)
    items_sorted = items[sort_idx]
    scores_sorted = scores[sort_idx]

    items = items_sorted
    scores = scores_sorted
    
    # Normalize scores for color mapping
    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    cmap = plt.cm.get_cmap('RdYlGn')
    colors = [cmap(v) for v in norm]

    fig, ax = plt.subplots(figsize=(5, 0.6 * len(items)))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(items))

    for i, (item, score, color) in enumerate(zip(items, scores, colors)):
        y = len(items) - i - 1
        ax.add_patch(plt.Rectangle((0, y), 1, 1, color=color))
        ax.text(0.5, y + 0.5, f"{item}: {score:.2f}",
                ha='center', va='center', fontsize=12, weight='bold',
                color='black' if norm[i] < 0.6 else 'white')

    ax.set_title(title, fontsize=14, weight='bold', pad=10)
    plt.tight_layout()
    plt.show()