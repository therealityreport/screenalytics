"use client";

import * as React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useDocs, type DocEntry } from "./docs-provider";

interface TodoModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type StatusFilter = "all" | "in_progress" | "draft" | "outdated";

function getStatusVariant(
  status: string
): "default" | "success" | "warning" | "info" | "secondary" | "destructive" {
  switch (status) {
    case "in_progress":
      return "warning";
    case "draft":
      return "info";
    case "outdated":
      return "destructive";
    default:
      return "secondary";
  }
}

function DocCard({ doc }: { doc: DocEntry }) {
  return (
    <div className="rounded-lg border p-3 space-y-2 hover:bg-muted/50 transition-colors">
      <div className="flex items-start justify-between gap-2">
        <div className="space-y-1 min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Badge variant={getStatusVariant(doc.status)} className="text-xs shrink-0">
              {doc.status}
            </Badge>
            <span className="text-xs text-muted-foreground capitalize">
              {doc.type}
            </span>
          </div>
          <h4 className="font-medium text-sm truncate">{doc.title}</h4>
          <p className="text-xs text-muted-foreground truncate">{doc.path}</p>
        </div>
      </div>

      {doc.features.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {doc.features.map((feature) => (
            <Badge key={feature} variant="outline" className="text-xs">
              {feature}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

export function TodoModal({ open, onOpenChange }: TodoModalProps) {
  const { getTodoItems } = useDocs();
  const [statusFilter, setStatusFilter] = React.useState<StatusFilter>("all");

  const allTodos = getTodoItems();
  const filteredTodos =
    statusFilter === "all"
      ? allTodos
      : allTodos.filter((doc) => doc.status === statusFilter);

  // Count by status
  const counts = {
    in_progress: allTodos.filter((d) => d.status === "in_progress").length,
    draft: allTodos.filter((d) => d.status === "draft").length,
    outdated: allTodos.filter((d) => d.status === "outdated").length,
  };

  const filterButtons: { label: string; value: StatusFilter; count?: number }[] = [
    { label: "All", value: "all", count: allTodos.length },
    { label: "In Progress", value: "in_progress", count: counts.in_progress },
    { label: "Draft", value: "draft", count: counts.draft },
    { label: "Outdated", value: "outdated", count: counts.outdated },
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>TO-DO</DialogTitle>
        </DialogHeader>

        <div className="flex flex-wrap gap-2 pb-2 border-b">
          {filterButtons.map((btn) => (
            <Button
              key={btn.value}
              variant={statusFilter === btn.value ? "default" : "outline"}
              size="sm"
              onClick={() => setStatusFilter(btn.value)}
            >
              {btn.label}
              {btn.count !== undefined && (
                <span className="ml-1 text-xs opacity-70">({btn.count})</span>
              )}
            </Button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto space-y-2 pr-2">
          {filteredTodos.length === 0 ? (
            <p className="text-muted-foreground text-sm py-4 text-center">
              No items match the current filter.
            </p>
          ) : (
            <>
              <p className="text-sm text-muted-foreground">
                {filteredTodos.length} item{filteredTodos.length !== 1 ? "s" : ""}
              </p>
              {filteredTodos.map((doc) => (
                <DocCard key={doc.id} doc={doc} />
              ))}
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
