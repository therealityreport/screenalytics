"use client";

import * as React from "react";

// Types matching docs_catalog.json schema
export interface FeaturePhases {
  [key: string]: string;
}

export interface Feature {
  title: string;
  status: string;
  paths_expected: string[];
  phases: FeaturePhases;
  pending: string[];
}

export interface DocEntry {
  id: string;
  title: string;
  path: string;
  status: string;
  last_updated: string;
  type: string;
  tags: string[];
  features: string[];
  models: string[];
  jobs: string[];
  ui_surfaces_expected: string[];
}

export interface DocsCatalog {
  version: number;
  generated_at: string;
  features: Record<string, Feature>;
  docs: DocEntry[];
}

interface DocsContextValue {
  catalog: DocsCatalog | null;
  loading: boolean;
  error: string | null;
  getFeaturesByPageId: (pageId: string) => Feature[];
  getDocsByPageId: (pageId: string) => DocEntry[];
  getTodoItems: () => DocEntry[];
}

const DocsContext = React.createContext<DocsContextValue | null>(null);

interface DocsProviderProps {
  children: React.ReactNode;
  initialCatalog?: DocsCatalog;
}

export function DocsProvider({ children, initialCatalog }: DocsProviderProps) {
  const [catalog, setCatalog] = React.useState<DocsCatalog | null>(
    initialCatalog ?? null
  );
  const [loading, setLoading] = React.useState(!initialCatalog);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (initialCatalog) return;

    // Fetch catalog if not provided
    fetch("/data/docs_catalog.json")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to fetch docs catalog: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        setCatalog(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [initialCatalog]);

  const getFeaturesByPageId = React.useCallback(
    (pageId: string): Feature[] => {
      if (!catalog) return [];

      // Find docs that reference this page
      const relevantDocs = catalog.docs.filter((doc) =>
        doc.ui_surfaces_expected.includes(pageId)
      );

      // Collect unique feature keys from those docs
      const featureKeys = new Set<string>();
      relevantDocs.forEach((doc) => {
        doc.features.forEach((f) => featureKeys.add(f));
      });

      // Return feature objects
      return Array.from(featureKeys)
        .map((key) => catalog.features[key])
        .filter(Boolean);
    },
    [catalog]
  );

  const getDocsByPageId = React.useCallback(
    (pageId: string): DocEntry[] => {
      if (!catalog) return [];
      return catalog.docs.filter((doc) =>
        doc.ui_surfaces_expected.includes(pageId)
      );
    },
    [catalog]
  );

  const getTodoItems = React.useCallback((): DocEntry[] => {
    if (!catalog) return [];
    return catalog.docs.filter((doc) =>
      ["in_progress", "draft", "outdated"].includes(doc.status)
    );
  }, [catalog]);

  const value: DocsContextValue = {
    catalog,
    loading,
    error,
    getFeaturesByPageId,
    getDocsByPageId,
    getTodoItems,
  };

  return <DocsContext.Provider value={value}>{children}</DocsContext.Provider>;
}

export function useDocs() {
  const context = React.useContext(DocsContext);
  if (!context) {
    throw new Error("useDocs must be used within a DocsProvider");
  }
  return context;
}
