"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import { FeaturesModal } from "./features-modal";
import { TodoModal } from "./todo-modal";
import { useDocs } from "./docs-provider";

interface PageHeaderProps {
  pageId: string;
  pageTitle: string;
  children?: React.ReactNode;
}

export function PageHeader({ pageId, pageTitle, children }: PageHeaderProps) {
  const [featuresOpen, setFeaturesOpen] = React.useState(false);
  const [todoOpen, setTodoOpen] = React.useState(false);
  const { getTodoItems, getFeaturesByPageId } = useDocs();

  const todoCount = getTodoItems().length;
  const featureCount = getFeaturesByPageId(pageId).length;

  return (
    <div className="flex items-center justify-between pb-4 border-b mb-6">
      <div className="flex items-center gap-4">
        <h1 className="text-2xl font-bold">{pageTitle}</h1>
        {children}
      </div>

      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setFeaturesOpen(true)}
        >
          PAGE FEATURES
          {featureCount > 0 && (
            <span className="ml-1 text-xs bg-primary/10 text-primary px-1.5 py-0.5 rounded">
              {featureCount}
            </span>
          )}
        </Button>

        <Button variant="outline" size="sm" onClick={() => setTodoOpen(true)}>
          TO-DO
          {todoCount > 0 && (
            <span className="ml-1 text-xs bg-yellow-100 text-yellow-800 px-1.5 py-0.5 rounded">
              {todoCount}
            </span>
          )}
        </Button>
      </div>

      <FeaturesModal
        open={featuresOpen}
        onOpenChange={setFeaturesOpen}
        pageId={pageId}
        pageTitle={pageTitle}
      />

      <TodoModal open={todoOpen} onOpenChange={setTodoOpen} />
    </div>
  );
}
