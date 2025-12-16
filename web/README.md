# Screenalytics Web UI

React/Next.js workspace UI for Screenalytics - face tracking and screen time analytics for reality TV.

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Environment Variables

Create a `.env.local` file:

```bash
# API base URL (default: http://localhost:8000)
NEXT_PUBLIC_API_BASE=http://localhost:8000

# Enable MSW mocking (for development without backend)
NEXT_PUBLIC_MSW=0
```

## OpenAPI Type Generation

TypeScript types are generated from the FastAPI OpenAPI schema.

### Where types come from

- **Source:** FastAPI server at `http://localhost:8000/openapi.json`
- **Generated to:** `src/api/generated/schema.ts`
- **Hand-written types:** `api/types.ts` (for client-side extensions)

### How to regenerate

```bash
# Ensure FastAPI is running on localhost:8000
npm run gen:api

# Or specify a different URL
OPENAPI_URL=https://your-api.example.com/openapi.json npm run gen:api
```

### When to regenerate

- After API schema changes (new endpoints, changed response shapes)
- Before starting work on a new feature that uses new API endpoints
- The generated file should be committed to the repo

### Files overview

| File | Purpose | Edit? |
|------|---------|-------|
| `src/api/generated/schema.ts` | Auto-generated OpenAPI types | **Never** |
| `api/types.ts` | Hand-written client types | Yes |
| `api/client.ts` | API fetch wrapper | Yes |
| `api/hooks.ts` | React Query hooks | Yes |

## Directory Structure

```
web/
├── app/
│   ├── layout.tsx          # Root layout (Screenalytics branding)
│   ├── page.tsx            # Redirects to /screenalytics/upload
│   └── screenalytics/      # Main app routes
│       ├── layout.tsx      # App layout with sidebar
│       ├── upload/         # Upload page
│       ├── episodes/       # Episode detail
│       └── faces/          # Faces review (stub)
├── api/                    # Hand-written API client code
├── src/api/generated/      # Auto-generated OpenAPI types
├── components/             # Shared components
├── lib/                    # Utilities and state machines
└── mocks/                  # MSW handlers for dev
```

## Tech Stack

- **Next.js 15** - React framework
- **React 19** - UI library
- **React Query** - Data fetching and caching
- **Radix UI** - Accessible primitives (Dialog, Toast)
- **CSS Modules** - Scoped styling
- **TypeScript** - Type safety
- **openapi-typescript** - Generate types from OpenAPI spec
