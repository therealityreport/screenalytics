"use client";

import * as React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { useDocs, type Feature, type DocEntry } from "./docs-provider";

interface FeaturesModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  pageId: string;
  pageTitle: string;
}

function getStatusVariant(
  status: string
): "default" | "success" | "warning" | "info" | "secondary" {
  switch (status) {
    case "complete":
      return "success";
    case "in_progress":
    case "partial":
      return "warning";
    case "not_started":
    case "scaffold":
    case "scaffold_only":
      return "secondary";
    default:
      return "info";
  }
}

function FeatureCard({ feature }: { feature: Feature }) {
  return (
    <div className="rounded-lg border p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-medium">{feature.title}</h4>
        <Badge variant={getStatusVariant(feature.status)}>{feature.status}</Badge>
      </div>

      {Object.keys(feature.phases).length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground font-medium">Phases:</p>
          <div className="space-y-1">
            {Object.entries(feature.phases).map(([phaseName, phaseStatus]) => (
              <div
                key={phaseName}
                className="flex items-center justify-between text-sm pl-2"
              >
                <span className="text-muted-foreground">{phaseName}</span>
                <Badge variant={getStatusVariant(phaseStatus)} className="text-xs">
                  {phaseStatus}
                </Badge>
              </div>
            ))}
          </div>
        </div>
      )}

      {feature.pending.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground font-medium">Pending:</p>
          <ul className="text-sm text-muted-foreground pl-4 list-disc">
            {feature.pending.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function FeaturesModal({
  open,
  onOpenChange,
  pageId,
  pageTitle,
}: FeaturesModalProps) {
  const { getFeaturesByPageId, getDocsByPageId } = useDocs();

  const features = getFeaturesByPageId(pageId);
  const docs = getDocsByPageId(pageId);

  // Extract unique models and jobs from docs
  const models = [...new Set(docs.flatMap((d) => d.models))];
  const jobs = [...new Set(docs.flatMap((d) => d.jobs))];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>PAGE FEATURES: {pageTitle}</DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="features" className="flex-1 overflow-hidden flex flex-col">
          <TabsList className="w-full justify-start">
            <TabsTrigger value="features">
              Features ({features.length})
            </TabsTrigger>
            <TabsTrigger value="models">Models ({models.length})</TabsTrigger>
            <TabsTrigger value="jobs">Jobs ({jobs.length})</TabsTrigger>
          </TabsList>

          <TabsContent
            value="features"
            className="flex-1 overflow-y-auto space-y-3 pr-2"
          >
            {features.length === 0 ? (
              <p className="text-muted-foreground text-sm">
                No features associated with this page.
              </p>
            ) : (
              features.map((feature) => (
                <FeatureCard key={feature.title} feature={feature} />
              ))
            )}
          </TabsContent>

          <TabsContent
            value="models"
            className="flex-1 overflow-y-auto space-y-2"
          >
            {models.length === 0 ? (
              <p className="text-muted-foreground text-sm">
                No models associated with this page.
              </p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {models.map((model) => (
                  <Badge key={model} variant="outline" className="text-sm">
                    {model}
                  </Badge>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="jobs" className="flex-1 overflow-y-auto space-y-2">
            {jobs.length === 0 ? (
              <p className="text-muted-foreground text-sm">
                No jobs associated with this page.
              </p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {jobs.map((job) => (
                  <Badge key={job} variant="outline" className="text-sm">
                    {job}
                  </Badge>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
