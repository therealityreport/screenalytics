# Faces Review UX Skill

Use this skill when modifying Faces Review or Smart Suggestions UI.

## When to Use

- Adding/modifying metrics display
- Changing cluster/track visualization
- Updating suggestion workflow
- Fixing UI bugs in review pages

## Key UI Patterns

### Similarity Badges

Always use `similarity_badges.py` for consistent color coding:

```python
from similarity_badges import render_similarity_badge

# Renders colored badge based on value
render_similarity_badge(
    value=0.85,
    metric_type="identity",  # or "cast", "track", "ambiguity"
    show_label=True
)
```

Color thresholds (identity/cast):
- Green (good): ≥ 0.75
- Yellow (warning): 0.65 - 0.75
- Red (bad): < 0.65

### Metrics Strip

Use `metrics_strip.py` for horizontal metric displays:

```python
from metrics_strip import render_metrics_strip

render_metrics_strip(cluster, metrics=[
    "identity_similarity",
    "cast_similarity",
    "ambiguity",
    "track_count"
])
```

### Confirmation Dialogs

Two-step confirmation for destructive actions:

```python
def _confirm_action(action_key: str, label: str, count: int) -> bool:
    confirm_key = f"confirm:{action_key}"

    if st.session_state.get(confirm_key):
        st.warning(f"Confirm {label}? Affects {count} items.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", key=f"yes_{action_key}"):
                st.session_state[confirm_key] = False
                return True
        with col2:
            if st.button("Cancel", key=f"no_{action_key}"):
                st.session_state[confirm_key] = False
        return False
    return False
```

### Undo History

For recoverable operations:

```python
# Store before state
if "undo_stack" not in st.session_state:
    st.session_state["undo_stack"] = []

# Before mutation
st.session_state["undo_stack"].append({
    "action": "unlink_cast",
    "data": current_state
})

# Undo button
if st.session_state["undo_stack"]:
    if st.button("Undo"):
        last = st.session_state["undo_stack"].pop()
        restore(last["data"])
```

### Cache Invalidation

After mutations, clear relevant caches:

```python
def invalidate_suggestions_cache(ep_id: str):
    keys = [
        f"cast_suggestions:{ep_id}",
        f"rescued_clusters:{ep_id}",
        f"temporal_only_clusters:{ep_id}",
    ]
    for key in keys:
        st.session_state.pop(key, None)
    st.cache_data.clear()
```

## Files to Check

| File | Purpose |
|------|---------|
| `pages/3_Faces_Review.py` | Main review page |
| `pages/3_Smart_Suggestions.py` | AI suggestions page |
| `similarity_badges.py` | Badge rendering |
| `metrics_strip.py` | Metrics display |
| `ui_helpers.py` | Common helpers |

## Widget Key Rules

Avoid duplicate ID errors:

```python
# Always include unique identifier
for i, cluster in enumerate(clusters):
    st.button("View", key=f"view_{cluster['id']}_{i}")

# For nested loops
for i, cluster in enumerate(clusters):
    for j, track in enumerate(cluster["tracks"]):
        st.checkbox("Select", key=f"sel_{cluster['id']}_{track['id']}_{i}_{j}")
```

## Session State Namespacing

```python
# Namespace by episode
ep_id = st.query_params.get("ep_id")
st.session_state[f"{ep_id}::filter_cast"]
st.session_state[f"{ep_id}::selected_clusters"]
```

## Testing Changes

```bash
# Syntax check
python -m py_compile apps/workspace-ui/pages/3_Faces_Review.py

# Run locally
streamlit run apps/workspace-ui/Upload_Video.py
```

## Checklist

- [ ] Uses similarity_badges for colors
- [ ] Uses metrics_strip for displays
- [ ] Confirmation for destructive actions
- [ ] Cache invalidation after mutations
- [ ] Unique widget keys
- [ ] Namespaced session state
