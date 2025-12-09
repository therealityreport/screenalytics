# Workspace UI Module - CLAUDE.md

Streamlit multi-page application for episode review and curation.

## Page Structure

Pages are prefixed with numbers for sidebar ordering:

| File | Page Name | Purpose |
|------|-----------|---------|
| `pages/0_Upload_Video.py` | Upload Video | Upload new episodes |
| `pages/1_Episodes.py` | Episodes | Episode list and status |
| `pages/2_Episode_Detail.py` | Episode Detail | Pipeline status, triggers |
| `pages/3_Faces_Review.py` | Faces Review | Manual cluster curation |
| `pages/3_Smart_Suggestions.py` | Smart Suggestions | AI-assisted cast linking |

## Shared Components

| File | Purpose |
|------|---------|
| `similarity_badges.py` | Color-coded metric badges |
| `metrics_strip.py` | Horizontal metrics display |
| `ui_helpers.py` | Common patterns (init_page, spinners) |
| `api_client.py` | Backend API wrappers |

## State Management

### Session State Namespacing

Always namespace keys by episode to avoid conflicts:

```python
# Good
st.session_state[f"{ep_id}::cluster_filter"]
st.session_state[f"{ep_id}::selected_tracks"]

# Bad - will conflict between episodes
st.session_state["cluster_filter"]
```

### Query Parameters

Use for shareable state:
```python
ep_id = st.query_params.get("ep_id")
```

### Cache Layers

```python
@st.cache_data(ttl=300)  # 5 minute cache
def load_suggestions(ep_id: str):
    ...
```

## UX Patterns

### Page Initialization

Always call first, before any other Streamlit calls:

```python
from ui_helpers import init_page

init_page(
    title="Faces Review",
    icon="👤",
    layout="wide"
)
```

### Heavy Operations

Wrap in spinner with error handling:

```python
with st.spinner("Loading clusters..."):
    try:
        data = load_heavy_data()
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.exception(e)
        return
```

### Confirmation Dialogs

Two-step for destructive actions:

```python
if st.button("Delete Cluster"):
    st.session_state["confirm_delete"] = True

if st.session_state.get("confirm_delete"):
    st.warning("Are you sure?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, delete"):
            do_delete()
            st.session_state["confirm_delete"] = False
    with col2:
        if st.button("Cancel"):
            st.session_state["confirm_delete"] = False
```

### Unique Widget Keys

Avoid duplicate ID errors:

```python
# Use index or unique ID in key
for i, cluster in enumerate(clusters):
    st.button("View", key=f"view_{cluster['id']}_{i}")
```

## Metrics Display

### Similarity Badges

```python
from similarity_badges import render_similarity_badge

render_similarity_badge(
    value=0.85,
    metric_type="identity",
    show_label=True
)
```

Color thresholds:
- Green: ≥0.75
- Yellow: 0.65-0.75
- Red: <0.65

### Metrics Strip

```python
from metrics_strip import render_metrics_strip

render_metrics_strip(cluster, metrics=[
    "identity_similarity",
    "cast_similarity",
    "ambiguity"
])
```

## Cache Invalidation

After mutations, clear relevant caches:

```python
def invalidate_episode_cache(ep_id: str):
    keys = [
        f"suggestions:{ep_id}",
        f"clusters:{ep_id}",
        f"cast_links:{ep_id}"
    ]
    for key in keys:
        st.session_state.pop(key, None)
    st.cache_data.clear()
```

## API Communication

Use `ApiResult` pattern:

```python
result = api_client.link_cast(ep_id, cluster_id, cast_id)
if result.success:
    st.success("Linked!")
    invalidate_episode_cache(ep_id)
else:
    st.error(f"Failed: {result.error}")
```

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Duplicate widget key | Same key used twice | Add unique suffix |
| Blank page | Exception before render | Add try/except, surface error |
| Stale data after action | Cache not invalidated | Call invalidation function |
| Widget state lost | Missing session state key | Use namespaced keys |

## Testing

```bash
# Syntax check
python -m py_compile apps/workspace-ui/pages/3_Faces_Review.py

# Run locally
streamlit run apps/workspace-ui/Upload_Video.py
```
