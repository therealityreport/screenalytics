"use client";

import { PageHeader } from "@/components/screenalytics/page-header";
import { Badge } from "@/components/ui/badge";
import { useDocs } from "@/components/screenalytics/docs-provider";

export default function DashboardPage() {
  const { catalog, loading, error, getTodoItems } = useDocs();

  if (loading) {
    return (
      <div className="p-6">
        <p className="text-muted-foreground">Loading docs catalog...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <p className="text-red-500">Error loading docs catalog: {error}</p>
      </div>
    );
  }

  const todoItems = getTodoItems();
  const features = catalog?.features ? Object.entries(catalog.features) : [];

  // Group todos by status
  const inProgress = todoItems.filter((d) => d.status === "in_progress");
  const drafts = todoItems.filter((d) => d.status === "draft");
  const outdated = todoItems.filter((d) => d.status === "outdated");

  return (
    <div>
      <PageHeader pageId="workspace-ui:docs_dashboard" pageTitle="Dashboard" />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Features Overview Card */}
        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <h3 className="font-semibold text-lg mb-4">Features Overview</h3>
          <div className="space-y-3">
            {features.map(([key, feature]) => (
              <div
                key={key}
                className="flex items-center justify-between text-sm"
              >
                <span>{feature.title}</span>
                <Badge
                  variant={
                    feature.status === "complete"
                      ? "success"
                      : feature.status === "partial" ||
                        feature.status === "in_progress"
                      ? "warning"
                      : "secondary"
                  }
                >
                  {feature.status}
                </Badge>
              </div>
            ))}
          </div>
        </div>

        {/* TO-DO Summary Card */}
        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <h3 className="font-semibold text-lg mb-4">TO-DO Summary</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">In Progress</span>
              <Badge variant="warning">{inProgress.length}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Drafts</span>
              <Badge variant="info">{drafts.length}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Outdated</span>
              <Badge variant="destructive">{outdated.length}</Badge>
            </div>
            <div className="border-t pt-3 mt-3">
              <div className="flex items-center justify-between font-medium">
                <span>Total</span>
                <span>{todoItems.length}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Stats Card */}
        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <h3 className="font-semibold text-lg mb-4">Documentation Stats</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Total Docs</span>
              <span className="font-medium">{catalog?.docs.length ?? 0}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Features</span>
              <span className="font-medium">{features.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">
                Catalog Version
              </span>
              <span className="font-medium">{catalog?.version ?? "-"}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Generated</span>
              <span className="font-medium text-xs">
                {catalog?.generated_at ?? "-"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent In Progress Items */}
      <div className="mt-8">
        <h3 className="font-semibold text-lg mb-4">Recent Work In Progress</h3>
        <div className="rounded-lg border bg-card shadow-sm divide-y">
          {inProgress.slice(0, 5).map((doc) => (
            <div key={doc.id} className="p-4 flex items-center justify-between">
              <div>
                <p className="font-medium text-sm">{doc.title}</p>
                <p className="text-xs text-muted-foreground">{doc.path}</p>
              </div>
              <div className="flex items-center gap-2">
                {doc.features.slice(0, 2).map((f) => (
                  <Badge key={f} variant="outline" className="text-xs">
                    {f}
                  </Badge>
                ))}
              </div>
            </div>
          ))}
          {inProgress.length === 0 && (
            <div className="p-4 text-sm text-muted-foreground">
              No items currently in progress.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
