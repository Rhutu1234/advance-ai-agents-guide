# Advanced AI Agents

*A deep-dive walkthrough of building agents that survive production — covering the ReAct architecture, planning and execution loops with replanning, tool selection at scale, tool schema design (including two-phase tools), state management with event sourcing, short- and long-term memory, workflow orchestration, durable agent execution with Azure Durable Functions, checkpointing and resumability, human approval workflows, multi-agent collaboration, supervisor-worker architectures, agent handoffs, communication protocols (MCP and A2A), failure recovery, infinite-loop prevention, budgets and iteration limits, and a decision framework for deterministic workflows versus autonomous agents.*

---

## Table of Contents

1. [Introduction](#introduction)
2. [ReAct Architecture](#1-react-architecture)
3. [Planning and Execution Loops](#2-planning-and-execution-loops)
4. [Tool Selection](#3-tool-selection)
5. [Tool Schema Design](#4-tool-schema-design)
6. [State Management](#5-state-management)
7. [Short-Term and Long-Term Memory](#6-short-term-and-long-term-memory)
8. [Workflow Orchestration](#7-workflow-orchestration)
9. [Durable Agent Execution](#8-durable-agent-execution)
10. [Checkpointing and Resumability](#9-checkpointing-and-resumability)
11. [Human Approval Workflows](#10-human-approval-workflows)
12. [Multi-Agent Collaboration](#11-multi-agent-collaboration)
13. [Supervisor-Worker Architecture](#12-supervisor-worker-architecture)
14. [Agent Handoffs](#13-agent-handoffs)
15. [Agent Communication Protocols](#14-agent-communication-protocols)
16. [Agent Failure Recovery](#15-agent-failure-recovery)
17. [Infinite-Loop Prevention](#16-infinite-loop-prevention)
18. [Budget and Iteration Limits](#17-budget-and-iteration-limits)
19. [Deterministic Workflows vs. Autonomous Agents](#18-deterministic-workflows-vs-autonomous-agents)
20. [Testing and Observing Advanced Agents](#19-testing-and-observing-advanced-agents)
21. [Putting It Together](#20-putting-it-together)
22. [Common Pitfalls](#21-common-pitfalls)
23. [Quick Reference Table](#quick-reference-table)
24. [Conclusion](#conclusion)

---

## Introduction

The earlier *AI Agents* guide in this series covered what an agent is: a model, a control loop, tools, and guardrails, with budgets, security, and evaluation. That is enough to build a demo that works. It is not enough to build an agent that **runs for twenty minutes without losing its place, survives a deployment in the middle of a task, waits three days for a manager's approval, coordinates five specialists without talking in circles, and never spends $400 chasing a bug in its own plan.** Those are the problems of *advanced* agents, and almost all of them turn out to be **distributed-systems and workflow problems wearing an AI costume**: state, durability, idempotency, retries, ordering, timeouts, and contracts between components.

```plaintext
DEMO AGENT                                       PRODUCTION AGENT
  one process, one request                         runs across crashes, deployments, and days
  state = the message list in memory               state = typed, versioned, persisted, replayable
  tools = a handful of functions                   tools = dozens, schema-validated, idempotent, permission-aware
  stop when the model says "done"                  stop on success criteria — OR budgets, loop detectors, approvals, or failure policy
  one agent                                        supervisors, workers, handoffs, typed contracts
  "retry if it breaks"                             classified failures, repair loops, compensation, escalation
```

The thesis of this guide, in one line: **put the intelligence in the model and the reliability in the code.** The model decides *what to try next*; deterministic code owns *state, limits, retries, persistence, permissions, and the shape of every message*. Section 18 turns that into a decision framework for when to use a fixed workflow, when to use an autonomous agent, and — most often — how to combine them.

This guide builds on *AI Agents*, *LLM Application Development*, *Semantic Kernel*, *Microsoft.Extensions.AI*, *AI Security*, *AI Observability*, and *AI Production / MLOps*; it avoids repeating them and goes deeper on structure and reliability. Code uses C#, `Microsoft.Extensions.AI` (`IChatClient`, `AIFunction`), and, for durability, Azure Durable Functions (Durable Task) alongside a framework-neutral checkpointing design you can host anywhere.

> **A note on frameworks and versions.** Agent frameworks are among the fastest-moving areas of the .NET AI stack (Semantic Kernel's agent and process frameworks, the Microsoft Agent Framework and its workflow/checkpointing features, hosted agent services, Durable Task, Temporal, Dapr Workflow, and Python-first tools like LangGraph). Protocol standards such as MCP and A2A are also evolving. The architecture and reliability principles are the durable part; verify API shapes, package names, and feature status against current documentation, and treat the code as templates. Thresholds and budgets are starting points to tune against your own evaluation set.

---

## 1. ReAct Architecture

### Reason, act, observe — repeat

**ReAct** (Reason + Act, from a 2022 research paper) is the pattern behind most tool-using agents: the model alternates between **reasoning** about what to do, **acting** by calling a tool, and **observing** the result, feeding the observation into the next round of reasoning, until it can answer.

```plaintext
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ THOUGHT      "I need the customer's last order before I can check the refund policy."│
 │ ACTION       get_orders(customer_id = 7721, limit = 1)                              │
 │ OBSERVATION  { order: 10482, total: 89.00, status: "delivered", delivered: "Oct 3" }│
 │ THOUGHT      "Delivered 12 days ago; the policy allows refunds within 30 days. I should confirm that policy."│
 │ ACTION       search_policy("refund window for delivered orders")                    │
 │ OBSERVATION  "[policy-4:2] Refunds are allowed within 30 days of delivery..."       │
 │ THOUGHT      "I have what I need."                                                  │
 │ ANSWER       "Yes — order 10482 is eligible for a refund until Nov 2. [policy-4:2]" │
 └──────────────────────────────────────────────────────────────────────────────┘
```

### Why it works

```plaintext
GROUNDING      Each step is anchored by a real observation instead of the model's guess — fewer hallucinated facts.
ADAPTIVITY     The next action depends on what the last one returned (a tool error, an empty result, a surprising value).
TRANSPARENCY   The trace shows what was tried and why — invaluable for debugging, audit, and evaluation.
GENERALITY     Any capability that can be wrapped as a tool becomes available to the same loop.
```

### Two ways to implement it

```plaintext
1. NATIVE TOOL CALLING (the modern default)
   The model API returns structured tool calls (name + JSON arguments); your loop executes them and returns tool results. The "thought" may be hidden (reasoning models), emitted as ordinary text before the call, or absent.
   + Reliable structure (schemas, parallel calls), less parsing fragility, supported across providers via IChatClient.
   - Reasoning content may not be exposed; behavior is shaped by the provider's tool-use training.

2. TEXT-FORMAT ReAct (for models without tool calling, or when you want explicit visible reasoning)
   The prompt teaches a format — "Thought: ... / Action: tool_name / Action Input: {json} / Observation: ..." — and YOUR code parses the model's text, runs the tool, appends an "Observation:", and calls the model again.
   + Works with any text model; fully inspectable.
   - Brittle parsing; the model must be stopped before it hallucinates an Observation; no native parallel calls.
```

### The native loop, with explicit step records

```csharp
public sealed record AgentStep(int Index, string? Thought, string? ToolName, string? ArgumentsJson, string? Observation, long Tokens, TimeSpan Duration);

public sealed class ReActAgent(IChatClient llm, IReadOnlyList<AIFunction> tools, IToolPolicy policy, RunBudget budget, LoopDetector loops)
{
    public async Task<AgentRunResult> RunAsync(string goal, string system, CancellationToken ct)
    {
        var messages = new List<ChatMessage> { new(ChatRole.System, system), new(ChatRole.User, goal) };
        var steps = new List<AgentStep>();
        var options = new ChatOptions { Tools = [.. tools], ToolMode = ChatToolMode.Auto, Temperature = 0.1f, MaxOutputTokens = 800 };

        for (int i = 1; ; i++)
        {
            budget.Charge(steps: 1);                                                       // throws BudgetExceededException when a hard limit is hit (Section 17)
            var sw = Stopwatch.StartNew();

            ChatResponse response = await llm.GetResponseAsync(messages, options, ct);    // THINK
            budget.Charge(tokens: response.Usage?.TotalTokenCount ?? 0);
            messages.AddRange(response.Messages);

            var calls = response.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>().ToList();
            if (calls.Count == 0)
                return AgentRunResult.Completed(response.Text, steps);                     // no action requested: the agent believes it is done

            foreach (FunctionCallContent call in calls)                                    // ACT + OBSERVE (through the policy gate)
            {
                loops.Observe(call.Name, call.Arguments);                                  // detect repeated identical calls (Section 16)
                string observation = await ExecuteAsync(call, ct);
                messages.Add(new ChatMessage(ChatRole.Tool, [new FunctionResultContent(call.CallId, observation)]));
                steps.Add(new AgentStep(i, response.Text, call.Name, JsonSerializer.Serialize(call.Arguments), Truncate(observation, 500), response.Usage?.TotalTokenCount ?? 0, sw.Elapsed));
            }
        }
    }
}
```

### A text-format ReAct parser (when you need one)

```csharp
// Expected model output (the prompt instructs: stop after "Action Input" and wait for the Observation):
//   Thought: I should look up the order.
//   Action: get_order
//   Action Input: {"orderNumber": 10482}
// or:
//   Thought: I now have enough information.
//   Final Answer: Order 10482 is eligible for a refund.
static readonly Regex ActionRx = new(@"Action:\s*(?<tool>[\w\-]+)\s*Action Input:\s*(?<input>\{.*\})", RegexOptions.Singleline | RegexOptions.Compiled);
static readonly Regex FinalRx  = new(@"Final Answer:\s*(?<answer>.+)$", RegexOptions.Singleline | RegexOptions.Compiled);

static ParsedTurn Parse(string text)
{
    if (FinalRx.Match(text) is { Success: true } f) return ParsedTurn.Final(f.Groups["answer"].Value.Trim());

    if (ActionRx.Match(text) is { Success: true } a)
    {
        // Parse and VALIDATE the JSON — the model's text is untrusted input. Reject anything that isn't an allow-listed tool with schema-valid arguments.
        return ParsedTurn.Action(a.Groups["tool"].Value, a.Groups["input"].Value);
    }
    return ParsedTurn.Malformed(text);                                                     // feed an error back ("Respond in the required format") — bounded retries (Section 15)
}
// Configure the model call with a STOP SEQUENCE of "Observation:" so it cannot invent the tool's result.
```

### Variants worth knowing

```plaintext
PLAN-AND-EXECUTE   Plan first, then execute (Section 2). More predictable than step-by-step improvisation.
ReWOO              "Reasoning WithOut Observation": the model writes the WHOLE tool plan upfront with placeholders for results (#E1, #E2), the system executes it, and the model is called again only to synthesize.
                   Far fewer model calls and tokens than ReAct, at the cost of adaptivity (the plan can't react to surprises).
REFLEXION          After a failed attempt, the agent writes a short self-critique ("I searched by name but the field needs an ID") that is carried into the next attempt as memory. Improves repeated-attempt tasks.
CodeAct            The agent's "actions" are snippets of code run in a sandbox, composing many operations per step. Powerful for data tasks; requires strict sandboxing (see AI Security, Section 9).
TREE SEARCH        Explore several candidate action sequences and pick the best (LATS-style); expensive, for hard reasoning/planning tasks.
```

### What ReAct does not give you

```plaintext
RELIABILITY       Nothing in the pattern prevents loops, runaway cost, wrong tools, or lost state — those are the rest of this guide.
TRUSTWORTHY THOUGHTS   A model's visible "reasoning" is a generated narrative, not a faithful record of its computation. Use it for debugging and UX, never as evidence that a decision was correct, and never as the security
                  boundary. Don't log or expose hidden reasoning content carelessly (it can contain sensitive material).
COST CONTROL      Every iteration re-sends a growing context; long ReAct runs get expensive and degrade as the context fills (see Advanced LLM Engineering, Section 5). Compact observations; cap steps.
```

---

## 2. Planning and Execution Loops

### Think about the whole task before touching anything

Pure ReAct decides one step at a time. For longer or riskier tasks, **plan-and-execute** separates concerns: a **planner** produces an explicit plan (data), **validation code** checks it, an **executor** runs the steps (mostly deterministically), and a **monitor** decides whether to continue, **replan**, or stop.

```plaintext
 GOAL ──> PLAN (model, structured output) ──> VALIDATE (code: tools allowed? DAG valid? within budget?) ──> [HUMAN REVIEW if sensitive]
              ^                                                                                             │
              │ replan (bounded)                                                                            v
              └─────────── MONITOR  <── OBSERVE <── EXECUTE steps (parallel where independent; model only where judgment is needed)
                              │
                              └─ goal met? ──> VERIFY against actual state ──> FINISH (or fail with a partial result)
```

### The plan as data: a dependency graph

```csharp
public sealed record PlanStep(
    string Id,                                   // "s1", "s2"...
    string Description,
    string Tool,                                 // must be an allow-listed tool (or a sub-agent role)
    JsonElement Arguments,                       // may reference earlier results with placeholders like "{{s1.result}}"
    IReadOnlyList<string> DependsOn);            // steps that must complete first

public sealed record Plan(string Goal, IReadOnlyList<PlanStep> Steps);

public sealed record StepResult(string StepId, bool Succeeded, string Output, string? Error);
```

### Validate the plan before running anything

```csharp
public static class PlanValidator
{
    public static IReadOnlyList<string> Validate(Plan plan, IReadOnlySet<string> allowedTools, int maxSteps)
    {
        var errors = new List<string>();
        var ids = plan.Steps.Select(s => s.Id).ToList();

        if (plan.Steps.Count == 0 || plan.Steps.Count > maxSteps) errors.Add($"Plan must have between 1 and {maxSteps} steps.");
        if (ids.Distinct().Count() != ids.Count)                  errors.Add("Step ids must be unique.");

        foreach (PlanStep s in plan.Steps)
        {
            if (!allowedTools.Contains(s.Tool))                   errors.Add($"{s.Id}: tool '{s.Tool}' is not allowed.");
            foreach (string dep in s.DependsOn)
                if (!ids.Contains(dep))                           errors.Add($"{s.Id}: unknown dependency '{dep}'.");
        }

        if (HasCycle(plan)) errors.Add("The plan contains a dependency cycle.");
        return errors;                                            // empty = valid; non-empty goes BACK to the planner as feedback (bounded retries)
    }

    private static bool HasCycle(Plan plan)
    {
        var state = new Dictionary<string, int>();                // 0 = unvisited, 1 = visiting, 2 = done
        var byId = plan.Steps.ToDictionary(s => s.Id);

        bool Visit(string id)
        {
            if (state.GetValueOrDefault(id) == 1) return true;    // back-edge: cycle
            if (state.GetValueOrDefault(id) == 2) return false;
            state[id] = 1;
            foreach (string dep in byId[id].DependsOn) if (byId.ContainsKey(dep) && Visit(dep)) return true;
            state[id] = 2;
            return false;
        }
        return plan.Steps.Any(s => Visit(s.Id));
    }
}
```

### Executing the DAG: parallel where independent

```csharp
public sealed class PlanExecutor(IToolRunner tools, int maxParallel = 4)
{
    public async Task<Dictionary<string, StepResult>> ExecuteAsync(Plan plan, RunBudget budget, CancellationToken ct)
    {
        var results = new Dictionary<string, StepResult>();
        var pending = plan.Steps.ToDictionary(s => s.Id);
        using var gate = new SemaphoreSlim(maxParallel);

        while (pending.Count > 0)
        {
            // Steps whose dependencies have all succeeded are READY; run them concurrently (bounded)
            var ready = pending.Values.Where(s => s.DependsOn.All(d => results.TryGetValue(d, out var r) && r.Succeeded)).ToList();

            if (ready.Count == 0)
                break;                                                    // blocked: a dependency failed -> let the monitor decide (replan or fail)

            var batch = await Task.WhenAll(ready.Select(async step =>
            {
                await gate.WaitAsync(ct);
                try
                {
                    budget.Charge(toolCalls: 1);
                    string args = Substitute(step.Arguments.GetRawText(), results);                  // fill {{s1.result}} placeholders (ReWOO-style), escaping values safely
                    return await tools.RunAsync(step.Id, step.Tool, args, ct);                       // returns a StepResult; tool exceptions become failed StepResults, not crashes
                }
                finally { gate.Release(); }
            }));

            foreach (StepResult r in batch) { results[r.StepId] = r; pending.Remove(r.StepId); }
        }
        return results;
    }
}
```

### The monitor and the replan loop

```csharp
public async Task<AgentRunResult> RunAsync(string goal, RunBudget budget, CancellationToken ct)
{
    const int MaxReplans = 2;
    Plan plan = await planner.CreatePlanAsync(goal, previousFailures: [], ct);

    for (int attempt = 0; ; attempt++)
    {
        IReadOnlyList<string> problems = PlanValidator.Validate(plan, allowedTools, maxSteps: 12);
        if (problems.Count > 0)
        {
            if (attempt >= MaxReplans) return AgentRunResult.Failed("No valid plan: " + string.Join("; ", problems));
            plan = await planner.CreatePlanAsync(goal, problems, ct);                         // feed the validation errors back
            continue;
        }

        if (RequiresApproval(plan)) await approvals.RequestPlanApprovalAsync(plan, ct);        // approve the PLAN when it contains sensitive steps (Section 10)

        var results = await executor.ExecuteAsync(plan, budget, ct);
        var failed = results.Values.Where(r => !r.Succeeded).ToList();

        if (failed.Count == 0 && results.Count == plan.Steps.Count)
            return await VerifyAndFinishAsync(goal, plan, results, ct);                        // check the ACTUAL end state, not just "steps ran"

        if (attempt >= MaxReplans) return AgentRunResult.Partial(results, failed);              // bounded: report what was done and what wasn't

        plan = await planner.CreatePlanAsync(goal, failed.Select(f => $"{f.StepId}: {f.Error}").ToList(), ct, completed: results);   // REPLAN around what already succeeded
    }
}
```

### When to plan, when not to

```plaintext
PLAN EXPLICITLY WHEN   the task has many steps, dependencies, parallelizable parts, side effects needing review, or a cost you want to estimate up front; or you need an auditable artifact ("here's what the agent intended").
IMPROVISE (ReAct) WHEN  steps depend heavily on what earlier observations reveal, the task is short, or exploration is the point.
HYBRID                  A coarse plan (3-6 milestones) with ReAct inside each milestone — structure at the top, adaptivity at the bottom.
PITFALLS                Plans go stale (the world changes mid-execution), models over-plan trivial tasks, and a bad early plan wastes the whole run — so validate, cap steps, replan with failure context, and verify outcomes.
VERIFY BY STATE         "All steps succeeded" is not "the goal is achieved." Check the real result (the ticket exists, the file validates) — ideally with a deterministic check.
```

---

## 3. Tool Selection

### Selection accuracy falls as the toolbox grows

With 5 well-named tools, models choose well. With 50, they confuse similar tools, miss relevant ones, and burn input tokens on definitions (every tool's schema is re-sent on every model call). Tool selection is a retrieval problem, a design problem, and a security problem.

```plaintext
SYMPTOMS OF TOO MANY / POORLY SEPARATED TOOLS
  - wrong tool chosen when two have overlapping descriptions            - the right tool never called (buried in a long list)
  - rising input-token cost per step (large tool definitions)           - arguments filled for a similar-but-different tool
  - model "inventing" tools that sound plausible
```

### Strategies, from simplest to most scalable

```plaintext
1. CURATE PER AGENT         Give each agent only the tools its role needs (a research agent has search and read tools — not email). Smaller sets, better selection, smaller blast radius.
2. ROUTE FIRST, THEN LOAD   A cheap classification step (rules, an embedding match, or a small model) picks the task category; load only that category's tools for the agent run.
3. TOOL RETRIEVAL           Embed every tool's name + description; per request (and per step), retrieve the top-k most relevant tools and expose only those, plus a small always-on core set.
4. HIERARCHICAL TOOLS       Expose a few coarse tools or a "find_tool"/"enable_tools" meta-tool; the agent discovers and activates specialized tools on demand. Powerful for hundreds of tools; adds a round trip.
5. SPECIALIST SUB-AGENTS    Wrap a family of tools behind a sub-agent exposed as ONE tool ("billing_agent"); the supervisor picks a specialist, the specialist picks the tool (Section 12).
6. PERMISSION-AWARE LISTS   Only tools the CURRENT USER may use are ever offered — the model can't call what it can't see, and the tool list isn't a leak of capabilities.
7. FORCE WHEN KNOWN         If the next action is certain (extract fields -> call one function), force it: ToolMode RequireAny / a specific tool, instead of letting the model choose.
```

### Tool retrieval in C#

```csharp
public sealed class ToolRetriever(IEmbeddingGenerator<string, Embedding<float>> embeddings)
{
    private readonly List<(AIFunction Tool, ReadOnlyMemory<float> Vector)> _index = [];

    public async Task IndexAsync(IEnumerable<AIFunction> tools, CancellationToken ct)
    {
        var list = tools.ToList();
        // Embed "name: description" — the same text the model will rely on. Re-index when tool descriptions change.
        var vectors = await embeddings.GenerateAsync(list.Select(t => $"{t.Name}: {t.Description}").ToList(), cancellationToken: ct);
        for (int i = 0; i < list.Count; i++) _index.Add((list[i], vectors[i].Vector));
    }

    public async Task<IReadOnlyList<AIFunction>> SelectAsync(string taskDescription, IReadOnlySet<string> allowedForUser, IEnumerable<AIFunction> alwaysOn, int topK, CancellationToken ct)
    {
        ReadOnlyMemory<float> q = await embeddings.GenerateVectorAsync(taskDescription, cancellationToken: ct);

        var selected = _index
            .Where(x => allowedForUser.Contains(x.Tool.Name))                                   // permissions FIRST: never offer a tool the user can't use
            .Select(x => (x.Tool, Score: TensorPrimitives.CosineSimilarity(q.Span, x.Vector.Span)))
            .OrderByDescending(x => x.Score)
            .Take(topK)
            .Select(x => x.Tool);

        return alwaysOn.Concat(selected).DistinctBy(t => t.Name).ToList();                       // a few core tools (ask_user, finish) are always present
    }
}

// Re-select between steps when the task evolves ("now I need to email the result"), but keep recently used tools in the set (hysteresis) so the toolbox doesn't flicker mid-run.
```

### Guiding the model's choice

```plaintext
NAME BY VERB AND OBJECT          search_tickets, get_order, create_refund_draft — consistent naming conventions across the toolset.
DESCRIBE WHEN (AND WHEN NOT)     "Use for X. Do NOT use for Y — use tool Z instead." Disambiguation text between neighbors is the single highest-value line you can write.
ELIMINATE OVERLAP                If two tools are hard to tell apart, merge them or sharpen the boundary. Ambiguity for you is ambiguity for the model.
EXAMPLES                         A short example call in the description helps for tricky argument shapes.
LIMIT PARALLELISM                Decide whether parallel calls are allowed (AllowMultipleToolCalls) — independent reads yes; side effects no.
CHEAP BEFORE EXPENSIVE           Nudge toward low-cost tools first ("prefer search_cache over search_web").
```

### Evaluate tool selection as its own component

```plaintext
Build a test set of (task, expected tool(s), expected arguments) and measure:
  TOOL-SELECTION ACCURACY   chosen tool == expected (confusion matrix shows which tools get confused with which)
  ARGUMENT ACCURACY          required arguments present and correct
  UNNECESSARY-CALL RATE      calls that didn't contribute
  MISSED-CALL RATE           tasks where a needed tool was never used
Use the confusion matrix to rewrite descriptions, merge tools, or introduce routing — before blaming the model. (See AI Agents, Section 11.)
```

---

## 4. Tool Schema Design

### The schema is the contract between a probabilistic caller and your deterministic code

Every tool definition does two jobs: it **teaches** the model how to call the tool (names, descriptions, parameter docs), and it **constrains** what valid calls look like (types, enums, required fields). Good schemas prevent whole classes of agent failure; bad ones produce hallucinated arguments that look plausible.

### Principles

```plaintext
FEW, SIMPLE PARAMETERS     Every optional parameter is another chance for a wrong value. Prefer 2-4 well-named parameters over a sprawling object.
ENUMS FOR CLOSED SETS      status: "open" | "closed" | "any" — not a free-text string the model must guess.
EXPLICIT TYPES AND FORMATS ISO 8601 dates ("2026-03-14"), ISO currency codes, units in the NAME or description (amountUsd, timeoutSeconds). Never leave units implicit.
DESCRIBE EVERY FIELD       What it is, valid values, an example, and what happens if omitted. The description is the model's only documentation.
ACCEPT THE ID THE MODEL HAS Models usually know names and have seen IDs in earlier tool results. Prefer parameters that match what the model can actually possess; resolve ambiguous names inside the tool (return candidates if several match).
AVOID DEEP NESTING AND UNIONS  Deeply nested objects, polymorphic unions, and recursive types reduce reliability and are poorly supported by strict-mode structured arguments.
REQUIRED vs OPTIONAL       Mark truly required fields required; give optional fields safe defaults documented in the description.
NO "ESCAPE HATCH" PARAMETERS  A free-form "query"/"sql"/"command"/"payload" parameter lets the model (or an injected prompt) do anything. Prefer typed parameters that code turns into parameterized operations.
ONE TOOL, ONE JOB          Split "manage_order(action, ...)" into get_order, cancel_order, update_shipping_address — each with a tight schema and its own permission and approval policy.
STABLE CONTRACTS           Version tools (get_order_v2) when the shape changes; in-flight durable runs may call the old shape (Section 9).
```

### A bad schema and a good one

```csharp
// ❌ Vague, stringly-typed, with an escape hatch
[Description("Manage orders")]
public Task<string> ManageOrder(string action, string data) { ... }

// ✅ Focused, typed, documented, safe defaults
public enum RefundReason { DamagedItem, NotAsDescribed, LateDelivery, Other }

[Description("Prepares a refund for ONE order and returns a preview plus a short-lived confirmation token. It does NOT move money. " +
             "Use only after confirming the order is delivered and within the refund window. For refund policy questions, use search_policy instead.")]
public async Task<PrepareRefundResult> PrepareRefundAsync(
    [Description("The customer-visible order number, e.g. 10482")] int orderNumber,
    [Description("Amount to refund in USD, e.g. 24.50. Must not exceed the order total.")] decimal amountUsd,
    [Description("The reason category. Use 'Other' only if none of the others fit.")] RefundReason reason,
    CancellationToken ct = default)
{ ... }
```

(Recent versions of the schema generator also reflect some validation attributes such as ranges and length limits into the schema — check what your `Microsoft.Extensions.AI` version supports — but **always validate again in code**.)

### Never trust the arguments: validate in the tool

```csharp
public static class ToolArgs
{
    // The model's arguments are untrusted input, exactly like a web request body. A schema nudges the model; code enforces.
    public static ValidationResult ValidateRefund(int orderNumber, decimal amountUsd)
    {
        if (orderNumber is < 1 or > 99_999_999)  return ValidationResult.Fail("orderNumber must be a positive order number.");
        if (amountUsd <= 0 || amountUsd > 10_000) return ValidationResult.Fail("amountUsd must be between 0.01 and 10,000.00.");
        if (decimal.Round(amountUsd, 2) != amountUsd) return ValidationResult.Fail("amountUsd must have at most 2 decimal places.");
        return ValidationResult.Ok();
    }
}

// On failure, return a HELPFUL error the model can act on — not an exception, not a stack trace:
return new ToolError("INVALID_ARGUMENT", "amountUsd must be between 0.01 and 10,000.00 (you sent 0).", Hint: "Check the order total with get_order and retry.");
```

### Design the output for the model, too

```plaintext
SMALL, STRUCTURED, USEFUL     Return only the fields the next decision needs, as compact JSON or short Markdown — not raw database rows or 5,000-line dumps (they consume context and distract).
STABLE ERROR CONTRACT         { "error": { "code": "NOT_FOUND" | "INVALID_ARGUMENT" | "NOT_PERMITTED" | "RATE_LIMITED" | "TRANSIENT", "message": "...", "hint": "what to try next", "retryable": true|false } }
                              The model (and your retry logic) can branch on code and retryable.
PAGINATION                    Return a page plus a cursor/next-page hint and a count ("showing 5 of 83; call again with cursor=...") instead of silently truncating.
TRUNCATION NOTES              If you cut output, say so: "[truncated: 12,400 more characters]".
NEXT-STEP HINTS               "To read a ticket in full, call get_ticket with its id." Cheap and effective.
NO SECRETS, NO INTERNALS      Never return credentials, internal hostnames, or stack traces — outputs re-enter the context and may reach users or logs (see AI Security, Section 9).
LABEL UNTRUSTED CONTENT       Wrap web pages, emails, and user-authored text as data ("<untrusted_content>...") — tool output is an injection vector.
```

### Two-phase tools for consequential actions

```csharp
// PHASE 1 — "prepare": validates everything, returns a human-readable PREVIEW and a short-lived token bound to these exact arguments, this user, and this run. Changes nothing.
[Description("Step 1 of 2. Validates a refund and returns a preview with a confirmation token. Does NOT move money.")]
public async Task<PrepareRefundResult> PrepareRefundAsync(int orderNumber, decimal amountUsd, RefundReason reason, CancellationToken ct)
{
    var order = await orders.GetForUserAsync(user.Id, orderNumber, ct);
    if (order is null) return PrepareRefundResult.NotFound();                                   // same message for "not yours" and "doesn't exist"
    if (ToolArgs.ValidateRefund(orderNumber, amountUsd) is { IsValid: false } bad) return PrepareRefundResult.Invalid(bad.Error);

    string token = await pending.CreateAsync(new PendingRefund(user.Id, runId, orderNumber, amountUsd, reason), ttl: TimeSpan.FromMinutes(15), ct);
    return new PrepareRefundResult($"Refund ${amountUsd:F2} to the original payment method for order {orderNumber} ({reason}).", token, RequiresApproval: amountUsd > AutoApproveLimit);
}

// PHASE 2 — "commit": executes ONLY what was prepared. The model cannot alter the amount between the phases; the approval (if required) is attached to the token.
[Description("Step 2 of 2. Executes a refund previously prepared with prepare_refund. Requires the confirmation token. May require human approval; if so it returns status 'awaiting_approval'.")]
public async Task<CommitRefundResult> CommitRefundAsync(string confirmationToken, CancellationToken ct)
{
    PendingRefund? p = await pending.ConsumeAsync(confirmationToken, user.Id, runId, ct);      // single use, bound to user + run, expires
    if (p is null) return CommitRefundResult.InvalidToken();

    if (p.AmountUsd > AutoApproveLimit && !await approvals.IsApprovedAsync(confirmationToken, ct))
        return CommitRefundResult.AwaitingApproval();

    await payments.RefundAsync(p.OrderNumber, p.AmountUsd, idempotencyKey: $"refund:{runId}:{p.OrderNumber}", ct);       // idempotent
    return CommitRefundResult.Done(p.OrderNumber, p.AmountUsd);
}
```

```plaintext
WHY TWO PHASES: the dangerous parameters are fixed at PREPARE time and validated; COMMIT takes only an opaque token, so a hijacked or confused model cannot change the amount, the order, or the account at execution time. It also gives
you a natural place for human approval (the preview is exactly what the reviewer sees) and makes retries safe (idempotency key + single-use token).
IDEMPOTENCY KEYS: every side-effecting tool accepts or derives one, so a retried or replayed call (Sections 8-9) can't charge, send, or create twice.
DRY-RUN FLAG: for less critical tools, a dryRun boolean that returns "what would happen" is a lightweight cousin of two-phase.
```

---

## 5. State Management

### Messages are not state

In a simple agent, "state" is the message list. In a durable, multi-step, multi-agent system that is dangerously insufficient: the transcript is a lossy, ever-growing, model-oriented artifact. **Agent state** is a *typed, versioned, explicit* record of everything the system needs to continue, audit, or recover a run — and the messages are *derived* from it for each model call.

```csharp
public sealed record AgentState(
    string RunId,
    int StateVersion,                                          // schema version, for migrating in-flight runs (Section 9)
    string Goal,
    RunStatus Status,                                          // Running | WaitingForApproval | Completed | Failed | Cancelled
    int StepIndex,
    Plan? Plan,                                                // the plan as data, with per-step status
    IReadOnlyDictionary<string, string> Facts,                 // verified facts gathered so far (with provenance elsewhere)
    IReadOnlyList<ArtifactRef> Artifacts,                      // references to large outputs stored in blob storage — NOT inlined
    IReadOnlyList<PendingApproval> PendingApprovals,
    IReadOnlyList<ToolCallRecord> ToolCalls,                   // each with id, args hash, status, result reference, idempotency key
    BudgetSnapshot Budget,
    IReadOnlyList<ChatMessage> Transcript);                    // the compacted conversation actually sent to the model
```

### What belongs in state

```plaintext
IDENTITY & CONTEXT     run id, user/tenant, goal, parent run (for sub-agents), trace context, model/prompt/tool-set versions PINNED for this run
PROGRESS               plan and step statuses, current step, completed work, partial results
EVIDENCE               facts found, with provenance (which tool call produced them) — so answers can cite and re-verification is possible
TOOL LEDGER            every tool call: id, arguments (or hash), status (pending/succeeded/failed), result reference, idempotency key — the basis of safe resume (Section 9)
PENDING EXTERNAL WORK  approvals awaiting decision, asynchronous jobs in flight, timers
BUDGETS                what has been consumed and what remains
CONVERSATION           the model-facing transcript — compacted and summarized, not authoritative
```

### Event sourcing: state as a fold over events

The most robust representation is an **append-only event log**. State is computed by folding events through a reducer. You get replay, audit, time travel, debugging, and resumption almost for free.

```csharp
public abstract record AgentEvent(string RunId, int Sequence, DateTimeOffset At);

public sealed record RunStarted(string RunId, int Sequence, DateTimeOffset At, string Goal, string UserId, PinnedVersions Versions) : AgentEvent(RunId, Sequence, At);
public sealed record ModelCalled(string RunId, int Sequence, DateTimeOffset At, long InputTokens, long OutputTokens, string ResponseRef) : AgentEvent(RunId, Sequence, At);
public sealed record ToolRequested(string RunId, int Sequence, DateTimeOffset At, string CallId, string Tool, string ArgumentsJson, string IdempotencyKey) : AgentEvent(RunId, Sequence, At);
public sealed record ToolReturned(string RunId, int Sequence, DateTimeOffset At, string CallId, bool Succeeded, string ResultRef, string? ErrorCode) : AgentEvent(RunId, Sequence, At);
public sealed record ApprovalRequested(string RunId, int Sequence, DateTimeOffset At, string ApprovalId, string Summary) : AgentEvent(RunId, Sequence, At);
public sealed record ApprovalDecided(string RunId, int Sequence, DateTimeOffset At, string ApprovalId, bool Approved, string DecidedBy) : AgentEvent(RunId, Sequence, At);
public sealed record HandoffOccurred(string RunId, int Sequence, DateTimeOffset At, string FromAgent, string ToAgent, string Reason) : AgentEvent(RunId, Sequence, At);
public sealed record RunCompleted(string RunId, int Sequence, DateTimeOffset At, string OutcomeRef) : AgentEvent(RunId, Sequence, At);
public sealed record RunFailed(string RunId, int Sequence, DateTimeOffset At, string Reason) : AgentEvent(RunId, Sequence, At);

public static class AgentReducer
{
    // Pure and deterministic: the same events always produce the same state
    public static AgentState Apply(AgentState state, AgentEvent e) => e switch
    {
        ToolRequested t   => state with { ToolCalls = [.. state.ToolCalls, new ToolCallRecord(t.CallId, t.Tool, t.ArgumentsJson, ToolStatus.Pending, null, t.IdempotencyKey)] },
        ToolReturned r    => state with { ToolCalls = state.ToolCalls.Select(c => c.CallId == r.CallId ? c with { Status = r.Succeeded ? ToolStatus.Succeeded : ToolStatus.Failed, ResultRef = r.ResultRef } : c).ToList() },
        ApprovalRequested a => state with { Status = RunStatus.WaitingForApproval, PendingApprovals = [.. state.PendingApprovals, new PendingApproval(a.ApprovalId, a.Summary)] },
        ApprovalDecided d => state with { Status = RunStatus.Running, PendingApprovals = state.PendingApprovals.Where(p => p.Id != d.ApprovalId).ToList() },
        RunCompleted      => state with { Status = RunStatus.Completed },
        RunFailed         => state with { Status = RunStatus.Failed },
        _                 => state
    };

    public static AgentState Replay(AgentState initial, IEnumerable<AgentEvent> events) => events.Aggregate(initial, Apply);
}
```

### The event store: append-only with optimistic concurrency

```csharp
public interface IAgentEventStore
{
    // Appends only if the stream's current version equals expectedVersion — prevents two workers from advancing the same run (split-brain)
    Task AppendAsync(string runId, long expectedVersion, IReadOnlyList<AgentEvent> events, CancellationToken ct);
    Task<IReadOnlyList<AgentEvent>> ReadAsync(string runId, long fromVersion, CancellationToken ct);
}
```

### State-management rules

```plaintext
SINGLE WRITER PER RUN       One worker advances a run at a time (a lease/lock, or optimistic concurrency on append). Two concurrent workers on one run duplicate side effects and corrupt state.
IDEMPOTENT EVENTS           Event IDs/sequence numbers make appends safe to retry.
LARGE DATA BY REFERENCE     Tool outputs, documents, and images live in blob storage; state and events hold references plus short summaries. State that carries megabytes makes every checkpoint slow and expensive.
IMMUTABLE UPDATES           Records with `with` expressions (or a reducer) — never mutate shared state in place; makes concurrency and replay reasoning tractable.
VERSIONED SCHEMAS           StateVersion on every state/event; "upcasters" translate old shapes to new so in-flight runs survive deployments.
SCOPE DELIBERATELY          RUN state (this task), SESSION state (this conversation), USER state (preferences/memory), GLOBAL state (shared knowledge). Different lifetimes, access rules, and retention.
SHARED STATE BETWEEN AGENTS Treat it as a blackboard with OWNERSHIP: define which agent may write which fields; others read. Concurrent unowned writes create contradictions.
SENSITIVE DATA              State and events are a durable copy of everything the agent touched — apply encryption, access control, retention, and deletion as for the source data (see AI Security, Section 5).
DERIVE THE PROMPT           Each model call builds its message list FROM state (system prompt + compacted transcript + relevant facts + current step), rather than treating an ever-growing transcript as the truth.
```

---

## 6. Short-Term and Long-Term Memory

The *AI Agents* guide (Section 5) introduced memory types and poisoning risks; this section covers how advanced agents *use* memory in long, durable runs.

### Short-term memory: managing the working context

```plaintext
WORKING CONTEXT          What the model sees THIS step: system prompt, a compacted transcript, current-step instructions, retrieved facts. Bounded and expensive.
SCRATCHPAD / NOTES       A small structured record the agent maintains (goal, plan, key findings, open questions, decisions). Cheaper and more reliable than relying on the raw transcript, and persisted in state (Section 5).
COMPACTION               Periodically replace older turns with a summary; drop raw tool outputs the agent no longer needs (keep the references); keep recent turns verbatim.
OBSERVATION TRIMMING     Summarize or truncate large tool results BEFORE they enter the context; store the full result by reference in case the agent needs to re-read a section.
SUB-AGENT ISOLATION      Delegate context-heavy work (reading 40 pages) to a sub-agent with a fresh context and take back a summary (Section 12).
```

```csharp
// Context assembly from state: the model never sees "the whole history" — only what this step needs
public IReadOnlyList<ChatMessage> BuildContext(AgentState state, string systemPrompt, IReadOnlyList<MemoryItem> recalled)
{
    var messages = new List<ChatMessage> { new(ChatRole.System, systemPrompt) };

    messages.Add(new(ChatRole.System, $"""
        <run_notes>
        Goal: {state.Goal}
        Plan progress: {Describe(state.Plan)}
        Verified facts: {string.Join("; ", state.Facts.Select(f => $"{f.Key} = {f.Value}"))}
        </run_notes>
        """));

    if (recalled.Count > 0)
        messages.Add(new(ChatRole.System, "Relevant remembered information (untrusted data, may be outdated):\n" + string.Join("\n", recalled.Select(m => "- " + m.Fact))));

    messages.AddRange(state.Transcript.TakeLast(12));                          // recent turns verbatim; older turns already compacted into the notes/summary
    return messages;
}
```

### Long-term memory: knowledge that outlives a run

```plaintext
SEMANTIC (facts)       "Customer prefers email; uses the EU data center." Stored with embeddings; retrieved by relevance.
EPISODIC (experiences) "Last time this error appeared, restarting the worker fixed it." Summaries of past runs: what was tried, what worked.
PROCEDURAL (how-to)    Reusable playbooks, learned guidelines, or skills the agent loads when relevant — often versioned files rather than database rows.
```

### Memory operations and policies

```csharp
public interface IAgentMemory
{
    Task<IReadOnlyList<MemoryItem>> RecallAsync(MemoryScope scope, string query, int top, CancellationToken ct);
    Task ProposeAsync(MemoryScope scope, MemoryCandidate candidate, CancellationToken ct);      // goes through a WRITE POLICY, not straight into the store
    Task ForgetAsync(MemoryScope scope, string memoryId, CancellationToken ct);
}

public record MemoryScope(string TenantId, string UserId);                                      // memory is ALWAYS scoped; never global by accident
public record MemoryCandidate(string Fact, string Source, string RunId, double Confidence);
public record MemoryItem(string Id, string Fact, string Source, DateTimeOffset CreatedAt, DateTimeOffset? ExpiresAt);

public sealed class MemoryWritePolicy
{
    public WriteDecision Evaluate(MemoryCandidate c) => c switch
    {
        _ when c.Source.StartsWith("untrusted:")  => WriteDecision.Reject,            // never persist facts derived from untrusted content automatically (memory poisoning)
        _ when ContainsSensitiveData(c.Fact)       => WriteDecision.Reject,            // no secrets/PII unless explicitly designed for
        _ when c.Confidence < 0.7                  => WriteDecision.Reject,
        _ when IsDuplicateOrConflicting(c)         => WriteDecision.QueueForReview,    // contradicting an existing memory needs resolution, not silent overwrite
        _                                           => WriteDecision.Accept
    };
}
```

### Memory design rules for advanced agents

```plaintext
WRITE DELIBERATELY      Memory writes are consequential (they influence all FUTURE runs). Gate them with a policy; consider human review for high-impact memories; record provenance (run, source, confidence).
SCOPE STRICTLY          Per user and tenant; test isolation (a cross-user memory leak is a data breach).
RETRIEVE, DON'T DUMP    Recall a few relevant items per step or run; don't inject "everything we know."
CONSOLIDATE             A background job merges duplicates, resolves conflicts (newest vs. highest-confidence vs. ask a human), summarizes episodes into durable lessons, and expires stale facts.
DECAY AND EXPIRE        Facts go stale; timestamp, TTL, and prefer recent. An outdated memory stated confidently is worse than none.
TREAT RECALLED MEMORY AS UNTRUSTED DATA   Delimit it, label it possibly outdated, and never let it carry instructions (see AI Security, Section 2).
USER CONTROL            Let users view, correct, and delete memories; support real erasure (including vector indexes and backups per policy).
EVALUATE                Measure memory's value (did recall change outcomes for the better?) and harm (wrong or poisoned recalls) — see AI Agents, Section 11.
MEMORY ≠ STATE          State is the run's authoritative working record; memory is cross-run knowledge. Keep them separate.
```

---

## 7. Workflow Orchestration

### Control flow as a graph

An advanced agent system is rarely one loop. It is a **workflow**: named steps (some deterministic code, some model calls, some tools, some sub-agents, some humans), connected by edges with conditions, carrying shared state. Writing the workflow as an explicit graph — rather than burying control flow inside prompts — makes it inspectable, testable, resumable, and bounded.

```plaintext
 [classify ticket (LLM)] ──> <category?> ──billing──> [retrieve policy] ──> [draft reply (LLM)] ──> [validate (code)] ──> <risky?> ──yes──> [human approval] ──> [send]
                                         │                                                                │no                                │
                                         ├──technical──> [diagnose agent (bounded ReAct)] ──────────────────┘                                  └──no──> [send]
                                         └──other──────> [route to human queue]
```

### Building blocks

```plaintext
NODE                 A unit of work: deterministic code, a model call, a tool call, a sub-agent, or a human task. Takes state, returns updated state.
EDGE                 A transition: unconditional, or conditional on state (a router function, possibly an LLM classifier whose output you validate against an allowed set).
PARALLEL / JOIN      Fan out independent nodes and join their results (bounded concurrency).
LOOP WITH EXIT       Refine/retry until a check passes — ALWAYS with a maximum iteration count (Section 16).
SUB-WORKFLOW         A node that is itself a workflow or a worker agent (hierarchical composition).
ERROR EDGES          Where to go when a node fails: retry, fallback node, compensation, human queue.
CHECKPOINT           State is persisted after each node (Section 9) so the workflow can resume.
```

### A compact typed workflow engine

```csharp
public sealed class Workflow<TState>
{
    private readonly Dictionary<string, Func<TState, CancellationToken, Task<TState>>> _nodes = new();
    private readonly Dictionary<string, Func<TState, string?>> _routes = new();          // next-node router per node (null = end)

    public Workflow<TState> Node(string name, Func<TState, CancellationToken, Task<TState>> work) { _nodes[name] = work; return this; }
    public Workflow<TState> Then(string from, string to) { _routes[from] = _ => to; return this; }
    public Workflow<TState> Route(string from, Func<TState, string?> router) { _routes[from] = router; return this; }

    public async Task<TState> RunAsync(string start, TState state, IWorkflowCheckpointer<TState> checkpoints, int maxSteps, CancellationToken ct)
    {
        string? current = start;

        for (int step = 0; current is not null; step++)
        {
            if (step >= maxSteps) throw new WorkflowLimitExceededException(maxSteps);        // HARD cap: no workflow runs forever

            state = await _nodes[current](state, ct);                                         // run the node
            string? next = _routes.TryGetValue(current, out var router) ? router(state) : null;

            if (next is not null && !_nodes.ContainsKey(next)) throw new InvalidOperationException($"Route from '{current}' returned unknown node '{next}'.");   // validate router output

            await checkpoints.SaveAsync(state, completedNode: current, nextNode: next, ct);  // CHECKPOINT after every node: (state, where to resume)
            current = next;
        }
        return state;
    }
}

// Resume after a crash: load the last checkpoint and continue from its nextNode
var (state, resumeAt) = await checkpoints.LoadLatestAsync(runId, ct);
await workflow.RunAsync(resumeAt, state, checkpoints, maxSteps: 50, ct);
```

### A support workflow with deterministic skeleton and agentic steps

```csharp
var workflow = new Workflow<TicketState>()
    .Node("classify", async (s, ct) => s with { Category = await classifier.ClassifyAsync(s.Text, ct) })               // model call; output validated against an allow-list
    .Route("classify", s => s.Category switch { "billing" => "retrieve_policy", "technical" => "diagnose", _ => "human_queue" })

    .Node("retrieve_policy", async (s, ct) => s with { Evidence = await retriever.SearchAsync(s.Text, s.User, ct) })    // deterministic retrieval
    .Then("retrieve_policy", "draft")

    .Node("diagnose", async (s, ct) => s with { Evidence = await diagnosticAgent.RunAsync(s, budget: s.Budget.Allocate(0.4), ct) })   // a BOUNDED agent inside the workflow
    .Then("diagnose", "draft")

    .Node("draft", async (s, ct) => s with { Draft = await drafter.DraftAsync(s, ct) })
    .Then("draft", "validate")

    .Node("validate", async (s, ct) => s with { Validation = validator.Check(s.Draft, s.Evidence) })                    // deterministic: citations valid? policy limits respected? PII-free?
    .Route("validate", s => !s.Validation.Passed ? "human_queue" : s.Validation.RequiresApproval ? "approval" : "send")

    .Node("approval", async (s, ct) => s with { Approved = await approvals.WaitAsync(s.RunId, s.Draft, ct) })            // durable wait (Sections 8, 10)
    .Route("approval", s => s.Approved ? "send" : "human_queue")

    .Node("send", async (s, ct) => { await mailer.SendAsync(s.User, s.Draft, idempotencyKey: s.RunId, ct); return s with { Done = true }; })
    .Node("human_queue", async (s, ct) => { await queue.EnqueueAsync(s, ct); return s with { Done = true }; });
```

### Engines and frameworks

```plaintext
HAND-ROLLED (as above)             Fine for small, in-process workflows; you own checkpointing and durability.
DURABLE TASK / AZURE DURABLE FUNCTIONS  Orchestrator code as ordinary C# with automatic persistence, replay, timers, and external events (Section 8). The workhorse for long-running agent workflows on Azure.
TEMPORAL, DAPR WORKFLOW            Durable-execution platforms with similar models and .NET SDKs.
SEMANTIC KERNEL PROCESS FRAMEWORK / MICROSOFT AGENT FRAMEWORK WORKFLOWS   Graph-style orchestration for agents with state passing and checkpointing features — evaluate their maturity and your hosting needs.
LOW-CODE (LOGIC APPS, POWER AUTOMATE)  Good for integration-heavy flows with agent steps inside.
PYTHON ECOSYSTEM (LANGGRAPH, ETC.)  Useful reference designs for graph-based agent state machines even if you build in .NET.
Choose by durability needs (minutes vs. days), hosting model, observability, team skills, and how well human-in-the-loop waits are supported.
```

---

## 8. Durable Agent Execution

### The problem: long-running work in an unreliable world

A real agent run may take minutes to days: model calls fail, tools time out, a deployment restarts the host, an approver is on vacation. An in-memory loop loses everything on any of these. **Durable execution** means the *workflow's progress is persisted automatically*, so a run resumes exactly where it left off after any interruption — with completed steps never re-executed.

### How durable execution works (Durable Task / Durable Functions)

```plaintext
ORCHESTRATOR FUNCTION   Your workflow logic, written as normal C# (loops, ifs, try/catch). It does NOT perform I/O itself; it SCHEDULES activities, timers, and waits for events.
ACTIVITY FUNCTIONS      The actual work: model calls, tool executions, database writes. Each activity's INPUT and RESULT are recorded in the orchestration's HISTORY.
HISTORY (EVENT LOG)     An append-only record of everything the orchestrator has done: "scheduled activity X", "activity X returned Y", "timer fired", "event received".
REPLAY                  When the orchestrator must continue (after a crash, a timer, an external event), the framework RE-RUNS the orchestrator code from the beginning against the history. Awaits on completed
                        activities return their RECORDED results instantly — so already-finished work is not repeated — until the code reaches the first not-yet-completed step, where real execution resumes.
```

### The determinism rules (the part everyone gets wrong)

Because the orchestrator is *replayed*, it must be **deterministic**: the same history must always lead the code down the same path.

```plaintext
❌ NEVER in orchestrator code:                         ✅ INSTEAD:
   DateTime.UtcNow / DateTime.Now                          context.CurrentUtcDateTime
   Guid.NewGuid() / Random                                 context.NewGuid() / pass values in as input
   direct I/O: HTTP calls, DB queries, file access         call an ACTIVITY
   MODEL CALLS (nondeterministic by nature!)               call an ACTIVITY — its result is recorded in history and REPLAYED, never re-generated
   Task.Delay / Thread.Sleep                               context.CreateTimer(...)
   reading mutable static/global state, environment        pass as input, or read inside an activity
   non-deterministic iteration (unordered collections)     deterministic ordering
   ordinary logging (duplicates on every replay)           context.CreateReplaySafeLogger(...)
THE KEY INSIGHT FOR AGENTS: the model is a nondeterministic function. Wrapping every model call in an ACTIVITY converts it into a recorded fact: on replay the agent does not "think again" and possibly diverge — it re-reads
what it decided last time. That is what makes agent runs resumable AND replayable.
```

### A durable agent orchestrator

```csharp
using Microsoft.DurableTask;
using Microsoft.Azure.Functions.Worker;

public record AgentRunInput(string RunId, string Goal, AgentState? State = null);
public record AgentDecision(DecisionKind Kind, string? ToolName, string? ArgumentsJson, string? FinalAnswer);
public enum DecisionKind { CallTool, RequestApproval, Finish, GiveUp }

public static class DurableAgent
{
    private const int MaxStepsPerInstance = 40;                       // history grows with every step; roll over to a fresh instance periodically (below)

    [Function(nameof(RunAgent))]
    public static async Task<AgentOutcome> RunAgent([OrchestrationTrigger] TaskOrchestrationContext context)
    {
        AgentRunInput input = context.GetInput<AgentRunInput>()!;
        ILogger log = context.CreateReplaySafeLogger(nameof(RunAgent));
        AgentState state = input.State ?? AgentState.Start(input.RunId, input.Goal);

        // Retries for TRANSIENT failures of activities (model 429s, tool timeouts) — configured declaratively, with backoff
        TaskOptions retry = TaskOptions.FromRetryPolicy(new RetryPolicy(maxNumberOfAttempts: 4, firstRetryInterval: TimeSpan.FromSeconds(5), backoffCoefficient: 2.0));

        for (int step = 0; step < MaxStepsPerInstance; step++)
        {
            if (state.Budget.Exhausted) return AgentOutcome.StoppedByBudget(state);                          // deterministic: budget lives in state, updated from recorded activity results

            // THINK: the model call is an ACTIVITY. Its decision is recorded; replay re-reads it instead of re-asking the model.
            AgentDecision decision = await context.CallActivityAsync<AgentDecision>(nameof(DecideActivity), state, retry);

            switch (decision.Kind)
            {
                case DecisionKind.CallTool:
                    // ACT: tool execution is an activity too, with an IDEMPOTENCY KEY derived from run id + step (safe if the activity is retried)
                    ToolResult result = await context.CallActivityAsync<ToolResult>(nameof(ExecuteToolActivity),
                        new ToolInvocation(state.RunId, step, decision.ToolName!, decision.ArgumentsJson!, IdempotencyKey: $"{state.RunId}:{step}"), retry);
                    state = state.WithToolResult(step, decision, result);
                    break;

                case DecisionKind.RequestApproval:
                    // HUMAN APPROVAL: durable wait — the instance consumes no compute while waiting hours or days, and survives restarts
                    await context.CallActivityAsync(nameof(NotifyApproverActivity), new ApprovalRequest(state.RunId, decision.ArgumentsJson!));
                    ApprovalDecision approval = await WaitForApprovalAsync(context, TimeSpan.FromHours(24));
                    state = state.WithApproval(approval);
                    break;

                case DecisionKind.Finish:
                    return AgentOutcome.Completed(decision.FinalAnswer!, state);

                default:
                    return AgentOutcome.GaveUp(state);
            }
        }

        // Long runs: restart this instance with the current state as input so the history doesn't grow without bound (the "eternal orchestration" pattern).
        context.ContinueAsNew(new AgentRunInput(state.RunId, state.Goal, state));
        return default!;                                              // unreachable in practice: ContinueAsNew ends this execution
    }

    private static async Task<ApprovalDecision> WaitForApprovalAsync(TaskOrchestrationContext context, TimeSpan timeout)
    {
        using var cts = new CancellationTokenSource();
        Task timer = context.CreateTimer(context.CurrentUtcDateTime.Add(timeout), cts.Token);           // durable timer, NOT Task.Delay
        Task<ApprovalDecision> approval = context.WaitForExternalEvent<ApprovalDecision>("ApprovalDecision");

        Task winner = await Task.WhenAny(approval, timer);
        if (winner == approval) { cts.Cancel(); return await approval; }
        return ApprovalDecision.Expired;                              // no decision in time: treat as NOT approved (fail closed)
    }

    // Activities: ordinary async code with real I/O. Their results are persisted in the orchestration history.
    [Function(nameof(DecideActivity))]
    public static async Task<AgentDecision> DecideActivity([ActivityTrigger] AgentState state, FunctionContext executionContext)
    {
        var brain = executionContext.InstanceServices.GetRequiredService<IAgentBrain>();               // wraps IChatClient + tools + policy
        return await brain.DecideAsync(state);
    }
}
```

```csharp
// Resolving an approval from outside (an API endpoint called by the approval UI, Teams card, or email link): raise the event the orchestration is waiting for
app.MapPost("/approvals/{runId}", async (string runId, ApprovalDecision decision, DurableTaskClient client, ClaimsPrincipal user) =>
{
    if (!await authz.CanApproveAsync(user, runId)) return Results.Forbid();
    await client.RaiseEventAsync(runId, "ApprovalDecision", decision with { DecidedBy = user.Identity!.Name! });
    return Results.Accepted();
});
```

### Fan-out workers with durable sub-orchestrations

```csharp
// Run specialist workers in parallel as sub-orchestrations: each is itself durable, with its own history, retries, and budget slice
var workerTasks = plan.Steps.Select(step =>
    context.CallSubOrchestratorAsync<WorkerResult>(nameof(RunWorker), new WorkerInput(state.RunId, step, state.Budget.Allocate(0.2)), retry));

WorkerResult[] results = await Task.WhenAll(workerTasks);        // a worker crash or deployment doesn't lose the others' progress
```

### Operating durable agents

```plaintext
VERSIONING IN-FLIGHT RUNS   A run started on Monday may still be executing after Wednesday's deployment. Replay requires the new orchestrator code to follow the SAME path as the recorded history, so breaking changes to
                            orchestrator logic break running instances. Mitigate: additive changes only; version orchestrations (RunAgent_v2) and let old instances drain; or deploy side-by-side and route new runs to the new version;
                            pin prompt/model/tool-set versions per run in state (Section 9).
HISTORY SIZE                Every step adds history; very long runs slow replay. Use ContinueAsNew (as above) and keep activity inputs/outputs small — pass references to blobs, not megabytes.
IDEMPOTENT ACTIVITIES       Activities have at-least-once semantics: a worker can crash after doing the work but before the result is recorded, causing a retry. Every side-effecting activity needs an idempotency key (Section 9).
PAYLOAD LIMITS              Inputs and outputs are serialized into history; respect size limits.
TIMEOUTS                    Give activities timeouts and orchestrations overall deadlines; a stuck model call shouldn't hold a run forever.
MANAGEMENT                  Query status, raise events (approvals), suspend/resume, terminate runaway instances (DurableTaskClient). A "kill this run" button is an essential safety control (see AI Security, Section 13).
OBSERVABILITY               Propagate the run id and trace context through activities; emit spans per step; replay-safe logging (see AI Observability).
COST                        Durable timers and waits are nearly free, but history storage and activity invocations are not; design steps at a sensible granularity.
OTHER ENGINES               Temporal, Dapr Workflow, and similar platforms provide the same model (deterministic workflow + recorded activities); the guidance carries over.
```

---

## 9. Checkpointing and Resumability

### Checkpointing without a durable-execution platform

You don't need Durable Functions to make an agent resumable. You need three things: **persisted state** (Section 5), a **write-ahead discipline** (record intent *before* acting and results *after*), and **idempotent side effects**. This section is the framework-neutral design — which is also what a durable-execution platform does for you.

### The core loop with checkpoints

```csharp
public sealed class ResumableAgent(IAgentEventStore events, IAgentBrain brain, IToolRunner tools, ILogger<ResumableAgent> log)
{
    public async Task<AgentState> RunOrResumeAsync(string runId, string? goalIfNew, CancellationToken ct)
    {
        // 1. LOAD: rebuild state from the event log (or start fresh)
        IReadOnlyList<AgentEvent> history = await events.ReadAsync(runId, fromVersion: 0, ct);
        AgentState state = history.Count == 0 ? await StartAsync(runId, goalIfNew!, ct) : AgentReducer.Replay(AgentState.Empty(runId), history);
        long version = history.Count;

        // 2. RECONCILE: any tool call recorded as REQUESTED but with no RETURN is "in doubt" — the process died mid-call. Resolve it BEFORE continuing.
        foreach (ToolCallRecord inDoubt in state.ToolCalls.Where(c => c.Status == ToolStatus.Pending).ToList())
            (state, version) = await ReconcileAsync(state, version, inDoubt, ct);

        // 3. CONTINUE from where we left off
        while (state.Status == RunStatus.Running)
        {
            AgentDecision decision = await brain.DecideAsync(state, ct);                                    // model call (its decision is recorded as an event)
            (state, version) = await AppendAsync(state, version, new ModelCalled(...), ct);

            if (decision.Kind == DecisionKind.Finish) { (state, version) = await AppendAsync(state, version, new RunCompleted(...), ct); break; }
            if (decision.Kind != DecisionKind.CallTool) { (state, version) = await AppendAsync(state, version, new RunFailed(...), ct); break; }

            string callId = Guid.NewGuid().ToString("N");
            string idempotencyKey = $"{runId}:{state.StepIndex}:{callId}";

            // WRITE-AHEAD: record the INTENT to call the tool BEFORE calling it
            (state, version) = await AppendAsync(state, version, new ToolRequested(runId, 0, DateTimeOffset.UtcNow, callId, decision.ToolName!, decision.ArgumentsJson!, idempotencyKey), ct);

            ToolResult result = await tools.RunAsync(decision.ToolName!, decision.ArgumentsJson!, idempotencyKey, ct);        // the side effect, protected by the idempotency key

            // RECORD the RESULT after
            (state, version) = await AppendAsync(state, version, new ToolReturned(runId, 0, DateTimeOffset.UtcNow, callId, result.Succeeded, result.Reference, result.ErrorCode), ct);
        }
        return state;
    }

    private async Task<(AgentState, long)> AppendAsync(AgentState state, long version, AgentEvent e, CancellationToken ct)
    {
        await events.AppendAsync(state.RunId, expectedVersion: version, [e], ct);          // optimistic concurrency: fails if another worker advanced this run
        return (AgentReducer.Apply(state, e), version + 1);
    }
}
```

### The crash windows — and how each is handled

```plaintext
CRASH POINT                                              WHAT'S RECORDED               ON RESUME
1. Before the model call                                  last completed event          call the model again (nothing was recorded; at worst you pay for one extra call)
2. After the model call, before ToolRequested is written  nothing about this decision   the decision is lost; the model is called again (it may decide differently — acceptable: no side effect occurred)
3. After ToolRequested, BEFORE the tool ran               "requested", no result        reconcile: the tool did not run -> run it (with the same idempotency key)
4. DURING the tool call                                   "requested", no result        reconcile: unknown whether it ran -> re-run with the SAME idempotency key; an idempotent tool returns the original outcome without repeating the effect
5. After the tool ran, BEFORE ToolReturned is written     "requested", no result        identical to 4: the idempotency key makes the re-run safe
6. After ToolReturned                                     result recorded               continue to the next model call
The only dangerous windows are 3-5 — and a single mechanism covers them: WRITE-AHEAD intent + an IDEMPOTENCY KEY enforced by the tool or the downstream system.
```

### Reconciling "in doubt" tool calls

```csharp
private async Task<(AgentState, long)> ReconcileAsync(AgentState state, long version, ToolCallRecord inDoubt, CancellationToken ct)
{
    // Best: ask the tool/downstream system whether the effect already happened for this idempotency key
    ToolOutcome? known = await tools.LookupOutcomeAsync(inDoubt.Tool, inDoubt.IdempotencyKey, ct);

    ToolResult result = known is not null
        ? known.ToResult()                                                                       // it DID happen: record the real result, don't repeat
        : inDoubt.IsSafeToRetry
            ? await tools.RunAsync(inDoubt.Tool, inDoubt.ArgumentsJson, inDoubt.IdempotencyKey, ct)      // idempotent or read-only: re-run with the same key
            : ToolResult.Unknown("The previous attempt's outcome could not be determined.");             // NOT idempotent and unknowable: do NOT guess

    if (result.IsUnknown)
        return await EscalateAsync(state, version, inDoubt, ct);                                  // route to a human / fail safe instead of risking a duplicate effect

    return await AppendAsync(state, version, new ToolReturned(state.RunId, 0, DateTimeOffset.UtcNow, inDoubt.CallId, result.Succeeded, result.Reference, result.ErrorCode), ct);
}
```

### Idempotency: the property that makes everything else safe

```plaintext
DEFINITION     Performing the operation twice with the same key has the same effect as performing it once.
HOW
  - The CALLER supplies a stable key (run id + step + call id); the TOOL passes it downstream (payment APIs, email services, and many REST APIs accept idempotency keys directly).
  - The TOOL/DOWNSTREAM stores key -> outcome and returns the stored outcome for repeats (a unique-key table works).
  - Naturally idempotent operations (PUT-style sets, "ensure X exists") need no key; "append"/"send"/"charge" operations do.
  - For read-only tools, retries are always safe.
  - For irreversible, non-idempotent, unknowable outcomes: don't guess — escalate (above).
AT-LEAST-ONCE vs EXACTLY-ONCE   Distributed systems give at-least-once delivery cheaply; "exactly-once effect" is achieved by at-least-once + idempotency. Design for that, not for the impossible.
```

### What to checkpoint, and when

```plaintext
WHEN      After every model call and after every tool result (the two expensive/irreversible boundaries). Granularity finer than that adds overhead without benefit; coarser risks redoing costly or unsafe work.
WHAT      Everything needed to continue: events/state (Section 5), step index, the tool ledger, pending approvals, budgets consumed, pinned versions (model, prompt, tool schemas), and references to stored artifacts.
WHERE     A durable store with optimistic concurrency: a SQL table of events, Cosmos DB, or an event-store service; blobs for large payloads. Redis alone is risky for state you cannot lose unless persistence is configured deliberately.
HOW SMALL Keep events compact (IDs, hashes, references, short summaries) so checkpoints stay cheap; snapshot state every N events to bound replay time.
SERIALIZE Use stable, versioned serialization (System.Text.Json with AIJsonUtilities.DefaultOptions for ChatMessage and AI content types); test round-trips.
```

### Resumability concerns beyond crashes

```plaintext
PINNED VERSIONS        A run that resumes after a deployment must keep using the prompt, model, and tool schemas it started with (recorded in state) — otherwise its behavior changes mid-run, and recorded tool calls
                       may no longer match the tool's current schema. Retire old versions only after in-flight runs drain.
STATE MIGRATIONS       When the state/event schema changes, upcast old events on read; never rewrite history in place.
LEASES AND LIVENESS    A worker holds a lease on a run while it advances it; a crashed worker's lease expires so another picks the run up. Avoid two workers advancing one run (the optimistic append guards the log; a lease avoids wasted work).
TIME TRAVEL AND FORKING  Because history is an append-only log, you can rebuild state at any point, inspect "what did the agent know at step 7," and FORK a run from an earlier checkpoint with edited state or a different model —
                       a powerful debugging and evaluation tool.
RETENTION AND PRIVACY  Event logs are a complete record of everything the agent saw and did: retain deliberately, encrypt, redact sensitive payloads (store references), and honor deletion requests (see AI Security, Section 5).
CANCELLATION           A cancel request is just another event; the loop checks for it between steps and runs compensation for completed side effects where needed (Section 15).
```

---

## 10. Human Approval Workflows

### Humans as first-class nodes in the workflow

The *AI Agents* guide introduced interrupt-and-resume. In advanced systems, approvals are **durable workflow steps** with their own lifecycle: they are requested, routed, reviewed, edited, approved or rejected, escalated, expired, and audited — and the agent must resume correctly whether the answer arrives in a minute or a week.

### What humans approve

```plaintext
ACTIONS     A specific tool call with exact arguments ("Refund $184.50 to order 10482"). The most common and the most precise.
PLANS       The whole plan before execution starts — efficient when many low-level actions follow from one decision.
OUTPUTS     A drafted email, report, or code change before it is sent or merged.
SPENDING    Budget extensions ("This task needs 3x the planned tokens — continue?").
ESCALATIONS Judgment calls the agent can't make (ambiguous goals, policy conflicts).
```

### Approval design

```plaintext
RISK-TIERED POLICY         Not everything needs a human. Classify actions by reversibility, blast radius, and value: auto-approve low-risk reads and small reversible writes; require approval above thresholds; require two approvers
                           for the highest tier. The policy is CODE (see AI Security, Section 9), versioned and testable.
SHOW THE REAL ACTION       The reviewer sees exactly what will happen (the preview from a two-phase tool), the evidence the agent used, and its stated rationale — not a vague "the agent wants to continue."
EDIT-AND-APPROVE           Let reviewers modify the proposal (change the amount, edit the draft) and approve the EDITED version; the agent resumes with the approved content, not its original.
BIND THE APPROVAL          The approval is tied to a specific proposal (token/hash). If the arguments change after approval, the approval is void — re-validate at execution time.
AUTHORIZED APPROVERS       The approver must be authorized for THIS action and THIS data (role, delegation, separation of duties — the requester can't approve their own request).
EXPIRY AND FAIL-CLOSED     Pending approvals expire (hours/days). On expiry the default is NOT approved. Never let timeout mean "proceed."
ESCALATION                 If no response within a window, escalate to a backup approver or a queue; record every hop.
MULTI-PARTY                N-of-M or sequential approvals for sensitive actions; the workflow waits for the quorum.
CHANNELS                   Meet people where they work — Teams/Slack adaptive cards, email with signed links, an internal task UI — each calling back an authenticated endpoint that raises the event.
AUDIT                      Who approved what, when, with which proposal hash, comments, and the agent's evidence — immutable (see AI Observability).
AVOID APPROVAL FATIGUE     If reviewers click "approve" reflexively, the control is theater. Keep gates for consequential actions, use bounds (caps, reversibility) so routine work flows, and track approval/rejection rates and time-to-decision.
RECORD FEEDBACK            Rejections and edits are labeled data: feed them into evaluation sets and policy tuning (see AI Agents, Section 11).
```

### The approval service

```csharp
public sealed record ApprovalRequest(string ApprovalId, string RunId, string ProposalHash, string Summary, string EvidenceRef, string[] RequiredRoles, int RequiredApprovals, DateTimeOffset ExpiresAt);
public sealed record ApprovalDecision(string ApprovalId, bool Approved, string DecidedBy, string? Comment, string? EditedProposalJson);

public sealed class ApprovalService(IApprovalStore store, INotifier notifier, IAuthorizationService authz, IDurableSignal signal)
{
    public async Task RequestAsync(ApprovalRequest request, CancellationToken ct)
    {
        await store.SaveAsync(request, ct);
        await notifier.NotifyApproversAsync(request, ct);                      // Teams card / email with a signed, single-use link
    }

    public async Task<Result> DecideAsync(ClaimsPrincipal approver, ApprovalDecision decision, CancellationToken ct)
    {
        ApprovalRequest? request = await store.GetAsync(decision.ApprovalId, ct);
        if (request is null || request.ExpiresAt < DateTimeOffset.UtcNow)         return Result.Expired();                // fail closed
        if (!await authz.CanApproveAsync(approver, request))                     return Result.Forbidden();               // authorized for THIS action
        if (approver.GetUserId() == await store.GetRequesterAsync(request, ct))  return Result.Forbidden();               // separation of duties

        await store.RecordDecisionAsync(decision, approver, ct);                  // audit trail first

        if (await store.QuorumReachedAsync(request, ct))                          // N-of-M
            await signal.RaiseAsync(request.RunId, "ApprovalDecision", await store.SummarizeDecisionAsync(request, ct), ct);   // wake the waiting run
        return Result.Ok();
    }
}
```

### On resume: re-validate before acting

```csharp
// The approval was granted hours or days ago. The world may have changed. Re-check preconditions before executing the approved action.
if (!await IsStillValidAsync(approvedProposal, ct))          // order cancelled? price changed? account closed? permissions revoked?
    return StepResult.Invalidated("Approved proposal no longer valid; returning to the agent with updated facts.");

// ...and verify the proposal hash matches what was approved
if (Hash(proposalToExecute) != approval.ProposalHash)
    return StepResult.Rejected("Proposal changed after approval.");
```

### Approval patterns in agent frameworks

```plaintext
FUNCTION-INVOCATION APPROVAL   Some frameworks (including recent Microsoft.Extensions.AI versions) offer approval-required functions: the loop PAUSES when a flagged tool is requested and surfaces an approval request to the caller,
                               who resumes with a decision. A convenient primitive — pair it with a durable store so the pause survives restarts.
POLICY GATE IN THE TOOL RUNNER  Your own interceptor returns "requires approval" as a tool result (the pattern in AI Agents), the run enters WaitingForApproval, and an external decision resumes it.
DURABLE EXTERNAL EVENTS         Durable Functions / Temporal: the workflow awaits an event with a timeout; an API call raises it (Section 8). The cleanest model for long waits.
PLAN-LEVEL APPROVAL            One review of the whole plan, then execution without interruption — plus runtime guards that stop the run if it deviates from the approved plan.
```

---

## 11. Multi-Agent Collaboration

### Why, and when, to add agents

Multiple agents are justified by **specialization** (focused prompts and toolsets), **context isolation** (each works with a lean context), **parallelism** (independent subtasks), **separation of duties** (a reviewer without write access), and **modularity**. They cost **latency, tokens, complexity, and new failure modes**. Start with one agent or a workflow; add agents when a single one *demonstrably* fails (measured, Section 19).

### Patterns recap, with the engineering concern each introduces

```plaintext
SUPERVISOR-WORKER      A coordinator decomposes, delegates, aggregates (Section 12).               Concern: the supervisor becomes a bottleneck and single point of failure; worker contracts must be tight.
PIPELINE               researcher -> writer -> reviewer in a fixed chain.                           Concern: errors propagate downstream; each stage needs validation of its input.
FAN-OUT / FAN-IN       Parallel workers, a merge step.                                              Concern: bounded concurrency, partial failures, reconciling conflicting outputs.
ROUTER / HANDOFF       A triage agent passes the conversation to a specialist (Section 13).         Concern: handoff loops, context loss, privilege boundaries.
CRITIC / REVIEWER      A generator and an evaluator iterate until criteria are met.                 Concern: infinite refinement loops; reviewer sycophancy; need objective acceptance criteria.
DEBATE / ENSEMBLE      Several agents propose, a judge selects.                                      Concern: cost multiplies; consensus isn't correctness.
BLACKBOARD             Agents read/write a shared structured state.                                  Concern: ownership and write conflicts (Section 5).
HIERARCHICAL           Supervisors of supervisors for large problems.                                Concern: depth multiplies latency, cost, and error; cap depth (Section 17).
```

### Principles for multi-agent systems

```plaintext
TYPED CONTRACTS, NOT CHATTER   Agents exchange structured messages (task specs and result records with schemas), not freeform conversation. Typed messages are validated, logged, tested, and secured; prose is none of those.
MINIMAL SHARED CONTEXT         Each agent receives only what it needs for its task and returns a compact result — "summaries up, tasks down." Don't forward full transcripts.
ONE OWNER PER RESPONSIBILITY   Every task, field, and decision has a single owning agent. Ambiguous ownership produces duplicated work and contradictions.
LEAST PRIVILEGE PER AGENT      A research agent can read; a payments agent can write — and only the latter sits behind approvals. Permissions are enforced in code by identity, not by each agent's prompt.
EXPLICIT TERMINATION           Each collaboration has success criteria and limits (rounds, budget). "Until they agree" is not a stopping condition.
TREAT AGENT OUTPUT AS UNTRUSTED INPUT   An injected instruction can hop from agent to agent. Validate every inter-agent message against its schema; never let one agent's text become another's instructions or tool permissions (see AI Security, Section 2).
INDEPENDENT FAILURE            A failing worker must not take down the run: timeouts, retries, partial results, reassignment (Section 15).
OBSERVABILITY ACROSS AGENTS    One trace per run with a span per agent task and message; correlation IDs through every hop (see AI Observability, Section 6).
MEASURE THE GAIN               Compare against a single-agent baseline on cost, latency, and quality; keep the multi-agent design only if it earns its complexity.
```

### Failure modes unique to multi-agent systems

```plaintext
HANDOFF PING-PONG        A hands to B, B hands back to A, forever. (Section 13)
COMPOUNDING ERRORS       An early agent's mistake is accepted as fact downstream; later agents elaborate on it. Validate at boundaries and carry provenance.
TELEPHONE GAME           Information degrades through summaries at every hop. Pass structured facts and references to evidence, not paraphrase.
SYCOPHANTIC REVIEW       A reviewer agent rubber-stamps the generator's work. Give reviewers objective checks (tests, validators, policies) and different instructions or models.
CONVERGENCE ON WRONG     Debating agents agree confidently. Ground in evidence and ground truth where possible.
COST EXPLOSION           N agents x M rounds x growing contexts. Budgets per agent and per run (Section 17).
CONFUSED AUTHORITY       A low-privilege agent persuades a high-privilege one to act (confused deputy). Authorization by identity at the tool, never by who "asked nicely."
```

---

## 12. Supervisor-Worker Architecture

### One agent plans and coordinates; specialists execute

The **supervisor** (also called orchestrator or manager) owns the goal. It decomposes work, assigns tasks to **workers** with narrow roles, tracks progress, validates results, handles failures, and synthesizes the final output. Workers own *how* a task is done within their tools and context; they never coordinate among themselves.

```plaintext
                              ┌────────────────────── SUPERVISOR ──────────────────────┐
   GOAL ───────────────────>  │ 1. decompose into typed tasks (validated)               │
                              │ 2. assign each to a worker role, with a budget slice    │
                              │ 3. run in parallel (bounded), track status              │
                              │ 4. validate each result against its contract            │
                              │ 5. retry / reassign / replan on failure                 │
                              │ 6. synthesize + verify the final answer                 │
                              └────┬─────────────────┬──────────────────┬───────────────┘
                                   v                 v                  v
                           RESEARCH WORKER     ANALYSIS WORKER     WRITING WORKER
                           (search + read)     (code sandbox)      (no external tools)
                           own context & tools own context & tools own context & tools
```

### Contracts between supervisor and workers

```csharp
public enum WorkerStatus { Succeeded, Failed, NeedsInput, BudgetExhausted }

// What the supervisor asks for
public sealed record WorkerTask(
    string TaskId,
    string Role,                                  // "research" | "analysis" | "writing" — maps to a worker with its own tools and permissions
    string Instruction,                           // self-contained: the worker does NOT see the whole conversation
    IReadOnlyDictionary<string, string> Inputs,   // facts and artifact REFERENCES from earlier tasks
    string ExpectedOutput,                        // what a good result looks like (and its schema)
    TimeSpan Deadline);

// What the worker returns — structured, with evidence, never a free-form chat transcript
public sealed record WorkerResult(
    string TaskId,
    WorkerStatus Status,
    string Summary,                               // concise result for the supervisor's context
    IReadOnlyDictionary<string, string> Artifacts,// references to large outputs stored elsewhere
    IReadOnlyList<string> Evidence,               // sources / tool call IDs supporting the claims
    string? Error,
    long TokensUsed);

public interface IWorker
{
    string Role { get; }
    Task<WorkerResult> ExecuteAsync(WorkerTask task, RunBudget budget, CancellationToken ct);
}
```

### The supervisor

```csharp
public sealed class Supervisor(IChatClient planner, IReadOnlyDictionary<string, IWorker> workers, IResultVerifier verifier, ILogger<Supervisor> log)
{
    public async Task<SupervisorOutcome> RunAsync(string goal, RunBudget budget, CancellationToken ct)
    {
        // 1. Decompose into typed tasks (structured output), then VALIDATE: known roles, sane count, acyclic dependencies
        TaskPlan plan = await DecomposeAsync(goal, ct);
        Validate(plan, workers.Keys, maxTasks: 8);

        var results = new Dictionary<string, WorkerResult>();
        using var gate = new SemaphoreSlim(3);                                              // bounded parallelism

        // 2-5. Execute in dependency order; run independent tasks concurrently, each with its OWN slice of the budget
        foreach (IReadOnlyList<PlannedTask> layer in plan.TopologicalLayers())
        {
            WorkerResult[] layerResults = await Task.WhenAll(layer.Select(async t =>
            {
                await gate.WaitAsync(ct);
                try   { return await RunWithRecoveryAsync(t, results, budget.Allocate(0.25), ct); }
                finally { gate.Release(); }
            }));

            foreach (WorkerResult r in layerResults) results[r.TaskId] = r;

            // A failed REQUIRED task that survived recovery attempts stops the run with a partial result rather than building on nothing
            if (layerResults.Any(r => r.Status != WorkerStatus.Succeeded && plan.IsRequired(r.TaskId)))
                return SupervisorOutcome.Partial(results, "A required task failed after recovery attempts.");
        }

        // 6. Synthesize from the VALIDATED results, then verify the final answer against evidence
        string answer = await SynthesizeAsync(goal, results, ct);
        VerificationResult check = await verifier.VerifyAsync(goal, answer, results, ct);
        return check.Passed ? SupervisorOutcome.Completed(answer, results) : SupervisorOutcome.Partial(results, check.Reason);
    }

    private async Task<WorkerResult> RunWithRecoveryAsync(PlannedTask task, IReadOnlyDictionary<string, WorkerResult> done, RunBudget budget, CancellationToken ct)
    {
        var workerTask = new WorkerTask(task.Id, task.Role, task.Instruction, ResolveInputs(task, done), task.ExpectedOutput, TimeSpan.FromMinutes(3));

        for (int attempt = 1; attempt <= 2; attempt++)                                       // bounded retries
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(workerTask.Deadline);                                              // per-task timeout

            try
            {
                WorkerResult result = await workers[task.Role].ExecuteAsync(workerTask, budget, cts.Token);
                if (result.Status == WorkerStatus.Succeeded && ResultContract.IsValid(result, task))   // validate the result at the boundary
                    return result;

                workerTask = workerTask with { Instruction = workerTask.Instruction + $"\nPrevious attempt problem: {result.Error ?? "result did not meet the contract"}" };   // feed the failure back
            }
            catch (OperationCanceledException) when (!ct.IsCancellationRequested) { log.LogWarning("Task {Task} timed out (attempt {Attempt})", task.Id, attempt); }
            catch (BudgetExceededException) { return Failed(task, WorkerStatus.BudgetExhausted, "Budget exhausted."); }
        }
        return Failed(task, WorkerStatus.Failed, "Task failed after retries.");
    }
}
```

### Workers: narrow, bounded agents

```csharp
public sealed class ResearchWorker(IChatClient llm, ResearchTools tools, ILogger<ResearchWorker> log) : IWorker
{
    public string Role => "research";

    public async Task<WorkerResult> ExecuteAsync(WorkerTask task, RunBudget budget, CancellationToken ct)
    {
        // A FRESH context: just the system prompt, the task, and its inputs — not the supervisor's conversation
        var agent = new ReActAgent(llm, tools.AsAIFunctions(), policy: ReadOnlyPolicy.Instance, budget, new LoopDetector());
        AgentRunResult run = await agent.RunAsync(task.Instruction, ResearchSystemPrompt, ct);

        return run.Status == "completed"
            ? new WorkerResult(task.TaskId, WorkerStatus.Succeeded, run.Answer!, run.ArtifactRefs, run.EvidenceIds, null, run.Tokens)
            : new WorkerResult(task.TaskId, WorkerStatus.Failed, "", new Dictionary<string, string>(), [], run.Reason, run.Tokens);
    }
}
```

### Design notes

```plaintext
SUPERVISOR AS CODE, NOT JUST A PROMPT   The decomposition is model-generated, but validation, ordering, concurrency, retries, budgets, and verification are deterministic code. A supervisor that is "just an LLM in a loop" inherits every
                                       agent failure mode at the top of the tree.
WORKER AS TOOL                          The simplest form: expose each worker as a function the supervisor agent calls (agents-as-tools). Easier, less control; the explicit supervisor above gives you parallelism, validation, and recovery.
SPECIALIZE BY TOOLS AND PERMISSIONS     Workers differ by what they can DO (tools, data access, approval policy), not just by persona text.
KEEP THE SUPERVISOR'S CONTEXT SMALL     It sees task summaries and result summaries, never workers' raw transcripts — otherwise the supervisor's context becomes the bottleneck.
DEPTH LIMITS                            Workers must not spawn unbounded sub-supervisors; cap delegation depth and total task count (Section 17).
CHEAPER MODELS FOR WORKERS              Route routine workers to small models and reserve a strong model for decomposition, verification, and synthesis (see Advanced LLM Engineering, Section 15).
REASSIGNMENT                            If a worker repeatedly fails, the supervisor can reassign to a different worker/model, simplify the task, or return a partial result.
PARTIAL RESULTS ARE FIRST-CLASS         Return what succeeded and what didn't, with reasons — often more useful (and honest) than failing everything.
```

---

## 13. Agent Handoffs

### Transferring ownership of a conversation

A **handoff** passes control of an interaction from one agent to another — a triage agent to billing, billing to a human, an agent to a more capable agent. Unlike supervisor delegation (the supervisor stays in charge), after a handoff the **receiving agent owns the thread**.

```plaintext
SUPERVISOR-WORKER     The supervisor keeps control and gets results back. "Do this subtask and report."
HANDOFF               Control TRANSFERS. The new agent talks to the user and decides what happens next. "You take it from here."
ESCALATION            A handoff to a more capable agent or a human when the current one is stuck, uncertain, or out of authority.
```

### Handoff as a tool

```csharp
public sealed record HandoffRequest(string TargetAgent, string Reason, string ContextSummary, IReadOnlyDictionary<string, string> Facts);

public sealed class HandoffTools(string currentAgent, IAgentDirectory directory)
{
    [Description("Transfers the conversation to a specialist agent. Use ONLY when the request is clearly outside your scope. " +
                 "Available: billing (payments, refunds, invoices), technical (bugs, errors, setup), human (complaints, legal, anything you cannot resolve).")]
    public HandoffSignal TransferTo(
        [Description("Target agent: billing | technical | human")] string target,
        [Description("One sentence: why this agent should take over")] string reason,
        [Description("A concise summary of the problem and what has been tried so far (the new agent will NOT see this conversation)")] string summary)
    {
        if (!directory.IsAllowedTarget(currentAgent, target)) return HandoffSignal.Rejected($"'{currentAgent}' cannot hand off to '{target}'.");   // topology is enforced in code
        return HandoffSignal.To(target, reason, summary);                                                                                         // a SIGNAL for the orchestrator, not an action
    }
}
```

### The orchestration loop that executes handoffs

```csharp
public sealed class HandoffOrchestrator(IAgentDirectory directory, ILogger<HandoffOrchestrator> log)
{
    private const int MaxHandoffs = 4;

    public async Task<ConversationResult> RunAsync(string userMessage, UserContext user, RunBudget budget, CancellationToken ct)
    {
        string currentAgent = "triage";
        var visited = new List<string> { currentAgent };
        HandoffRequest? pending = null;
        var context = new ConversationContext(user, userMessage);

        while (true)
        {
            // The receiving agent gets a COMPACT BRIEFING, not the whole transcript
            AgentTurnResult turn = await directory.Get(currentAgent).RunAsync(context.ForAgent(currentAgent, pending), budget, ct);

            if (turn.Handoff is null) return ConversationResult.Reply(turn.Reply, handledBy: currentAgent);

            // --- handoff handling (all deterministic code) ---
            if (visited.Count > MaxHandoffs)                                        return Escalate(context, "Too many handoffs.");        // cap
            if (turn.Handoff.TargetAgent == "human")                                return await HandToHumanAsync(context, turn.Handoff, ct);
            if (visited.Count(a => a == turn.Handoff.TargetAgent) >= 2)             return Escalate(context, "Handoff loop detected.");    // ping-pong guard

            log.LogInformation("Handoff {From} -> {To}: {Reason}", currentAgent, turn.Handoff.TargetAgent, turn.Handoff.Reason);

            pending = turn.Handoff;                                                 // the briefing for the next agent
            currentAgent = turn.Handoff.TargetAgent;
            visited.Add(currentAgent);
        }
    }
}
```

### Handoff design rules

```plaintext
EXPLICIT TOPOLOGY          Define who may hand off to whom (a directed graph) in code. Free-form "transfer to anyone" invites loops and privilege games.
COMPACT BRIEFING           Pass a concise summary, the verified facts, and references to evidence — not the full transcript. The new agent should be able to act without re-asking the user for everything they already said.
STATE TRANSFER             Carry structured state (customer id, order number, what was attempted and failed) in a typed record; don't rely on prose summaries alone.
OWNERSHIP IS CLEAR         At any moment exactly one agent owns the thread; log HandoffOccurred events (Section 5).
PRIVILEGES DO NOT TRANSFER The receiving agent runs with ITS OWN tools and the USER's permissions — a handoff must never carry over the sender's elevated rights or grant new ones. Authorization is by identity at each tool.
LOOP GUARDS                A maximum number of handoffs per conversation, and detection of A -> B -> A patterns; on violation, ESCALATE to a human rather than continuing.
GRACEFUL HANDOFF TO HUMANS Package the briefing, evidence, and agent's attempts for the human queue; tell the user what happens next (expected wait, channel).
SAY WHEN IT HAPPENS        Tell the user they've been transferred (and to whom) — silent switches confuse and erode trust.
UNTRUSTED BRIEFING         The briefing is model-generated text derived from user input; the receiving agent treats it as data, not instructions (it can contain injected content).
RETURN PATH                Decide whether the specialist can hand BACK (and under what conditions) or only escalate onward.
EVALUATE                   Measure handoff accuracy (right specialist), unnecessary handoffs, loops, and information loss (did the specialist have to re-ask?).
```

---

## 14. Agent Communication Protocols

### Agents talk to tools, to each other, and to users — each with different needs

```plaintext
AGENT <-> TOOLS / DATA      "Here are functions and resources you can use."                 -> function calling, OpenAPI, and the MODEL CONTEXT PROTOCOL (MCP)
AGENT <-> AGENT             "Here is a task; report back / take over."                       -> in-process typed messages, queues, and the AGENT2AGENT (A2A) protocol for cross-system agents
AGENT <-> USER INTERFACE    "Stream tokens, show tool progress, ask for approval."           -> streaming events (SSE/WebSockets; emerging agent-UI protocols)
```

### Standards (verify current specs — they evolve)

```plaintext
MCP (Model Context Protocol)   A standard way for an agent host to connect to TOOL and DATA servers: tool discovery (names, descriptions, JSON schemas), invocation, resources, and prompts, over stdio or HTTP.
                               Gives agents a growing ecosystem of tools without bespoke integration. Security note: MCP servers are a supply-chain surface (tool poisoning, rug-pulls, confused-deputy token handling) —
                               allow-list, pin, review, scope tokens, and wrap tools in your policy layer (see AI Security, Section 9). In .NET, the C# MCP SDK exposes server tools as AIFunction-compatible objects.

A2A (Agent2Agent)              An open protocol for agents built on DIFFERENT frameworks/vendors to discover and collaborate: an "agent card" describing an agent's capabilities and endpoint; JSON-RPC over HTTP; a TASK abstraction
                               with a lifecycle (for example submitted -> working -> input-required -> completed / failed / canceled); streaming updates (SSE) and push notifications for long-running tasks; artifacts as results.
                               Relevant when agents cross organizational or framework boundaries; overkill inside one process.

OPENAPI / REST / gRPC          Ordinary service interfaces: generate tools from OpenAPI specs; use gRPC for typed, high-throughput internal agent services.
AGENT-UI PROTOCOLS             Emerging conventions for streaming agent events (messages, tool calls, state deltas, approval requests) to front-ends; evaluate maturity before adopting.
```

### An internal message envelope (typed, traceable, safe)

```csharp
// Every inter-agent message in YOUR system has the same envelope: routable, correlated, versioned, size-bounded, and schema-checked.
public sealed record AgentMessage<TPayload>(
    string MessageId,
    string RunId,                       // one run = one trace
    string CorrelationId,               // links a response to its request
    string From,
    string To,
    string Type,                        // "task.request" | "task.result" | "task.progress" | "handoff" | "approval.request" | "error"
    int SchemaVersion,
    TPayload Payload,
    DateTimeOffset SentAt,
    DateTimeOffset? ExpiresAt,          // stale work is dropped, not executed late
    string TraceParent);                // W3C trace context so one distributed trace spans every agent

public interface IAgentBus
{
    Task SendAsync<T>(AgentMessage<T> message, CancellationToken ct);
    IAsyncEnumerable<AgentMessage<JsonElement>> ReceiveAsync(string agent, CancellationToken ct);
}

// Receiving side: validate BEFORE any model sees the content
public static bool TryAccept(AgentMessage<JsonElement> msg, IMessageSchemaRegistry schemas, out string reason)
{
    if (msg.ExpiresAt < DateTimeOffset.UtcNow)                     { reason = "expired";  return false; }
    if (msg.Payload.GetRawText().Length > 64_000)                  { reason = "too large"; return false; }
    if (!schemas.Validate(msg.Type, msg.SchemaVersion, msg.Payload)) { reason = "schema violation"; return false; }
    reason = "";
    return true;
}
```

### Transport choices

```plaintext
IN-PROCESS CHANNELS      System.Threading.Channels between agents in one host: fastest, simplest; lost on crash (so checkpoint at the boundaries).
QUEUES / TOPICS          Azure Service Bus, RabbitMQ, Kafka: durable, decoupled, retry and dead-letter built in, natural for worker agents scaled by queue depth (KEDA). Add idempotent consumers (messages are at-least-once).
HTTP / gRPC              Request-response between agent services; add timeouts, retries, and idempotency keys; use streaming for progress.
SHARED STATE (BLACKBOARD)  Agents coordinate through a store with defined ownership; good for loosely coupled collaboration, harder to trace causality.
Prefer ASYNCHRONOUS, DURABLE messaging for work that takes more than a few seconds, and synchronous calls only for short, bounded requests.
```

### Protocol-level safety

```plaintext
AUTHENTICATE AGENTS      Agents are workloads: give each an identity (managed/workload identity), authenticate calls (OAuth, mTLS), and authorize which agent may send which message types to which agent.
VALIDATE EVERYTHING      Schema-check every message; reject unknown types and oversize payloads; version schemas.
UNTRUSTED CONTENT        A message from another agent — even an internal one — can carry attacker-influenced text. Delimit it as data; never let it modify the receiver's instructions, tools, or permissions.
NO AMBIENT AUTHORITY     Messages carry the originating USER's identity/claims (propagated and verified), never "trust me, I'm the supervisor" authority.
BOUND THE CHATTER        Maximum messages per run, per agent pair, and per time window — runaway agent-to-agent loops are denial-of-wallet attacks (Sections 16-17).
AUDIT                    Log every message's metadata (not necessarily content) with run and trace IDs.
CROSS-ORGANIZATION (A2A) Treat remote agents as untrusted third parties: scoped capabilities, data minimization, contractual and technical limits on what is shared and executed.
```

---

## 15. Agent Failure Recovery

### Everything fails: plan the recovery, not just the happy path

Agents fail in more ways than ordinary services, because they have a probabilistic component making decisions and many dependencies executing side effects. **Classify failures, because each class has a different remedy.**

### A failure taxonomy

```plaintext
CLASS                         EXAMPLES                                                             REMEDY
TRANSIENT INFRASTRUCTURE      model 429/5xx/timeouts, network blips, tool unavailable                retry with jittered backoff; circuit breaker; fall back to another deployment/model
INVALID MODEL OUTPUT          malformed JSON, schema violation, unknown tool name, bad arguments     REPAIR LOOP: feed the validation error back (bounded 1-2 attempts); then escalate/fail
TOOL-LEVEL ERRORS             not found, invalid argument, permission denied, rate limited           return a HELPFUL structured error; the model adapts; classify retryable vs. not
WRONG BEHAVIOR                wrong tool, wrong plan, hallucinated "success"                         verification against real state; replan; stronger model for that step; human review
LOOPS / NO PROGRESS           repeated calls, oscillation, endless refinement                        detection and intervention (Section 16)
BUDGET EXHAUSTION             steps, tokens, cost, time limits hit                                   graceful wrap-up with partial results (Section 17)
POLICY DENIALS                approval rejected, action not permitted                                tell the agent plainly; replan within allowed actions; or end
PARTIAL COMPLETION            some steps done, later one failed                                      checkpoint resume; COMPENSATE completed side effects if the whole must be atomic; return partial results
PROCESS / HOST FAILURE        crash, restart, deployment                                             durable state and idempotent resume (Sections 8-9)
CONTEXT PROBLEMS              overflow, poisoned or contradictory context, drift                     compaction; restart from a clean checkpoint with a summary; sub-agent isolation
POISONED INPUT                prompt injection in tool output or documents                           isolation, least privilege, quarantine pattern (see AI Security, Section 2)
```

### Retry classification and circuit breaking

```csharp
public enum FailureKind { Transient, InvalidInput, NotPermitted, NotFound, Permanent, Unknown }

public static class FailureClassifier
{
    public static FailureKind Classify(Exception ex) => ex switch
    {
        ClientResultException { Status: 429 or 408 or >= 500 }  => FailureKind.Transient,
        HttpRequestException or TimeoutException                  => FailureKind.Transient,
        ToolValidationException                                   => FailureKind.InvalidInput,
        UnauthorizedAccessException                               => FailureKind.NotPermitted,
        KeyNotFoundException                                      => FailureKind.NotFound,
        _                                                         => FailureKind.Unknown
    };
}

public sealed class ResilientToolInvoker(ICircuitBreakers breakers, ILogger<ResilientToolInvoker> log)
{
    public async Task<ToolResult> InvokeAsync(string tool, Func<CancellationToken, Task<ToolResult>> call, bool idempotent, CancellationToken ct)
    {
        if (breakers.IsOpen(tool))
            return ToolResult.Error("UNAVAILABLE", $"The '{tool}' tool is temporarily unavailable. Try another approach or report the limitation.", retryable: false);     // fail fast

        for (int attempt = 1; ; attempt++)
        {
            try
            {
                ToolResult result = await call(ct);
                breakers.RecordSuccess(tool);
                return result;
            }
            catch (Exception ex) when (!ct.IsCancellationRequested)
            {
                FailureKind kind = FailureClassifier.Classify(ex);
                breakers.RecordFailure(tool);

                // Retry ONLY transient failures, ONLY for idempotent calls (or with an idempotency key), with a hard cap and jitter
                if (kind == FailureKind.Transient && idempotent && attempt < 3)
                {
                    await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt - 1) + Random.Shared.NextDouble()), ct);
                    continue;
                }

                log.LogWarning(ex, "Tool {Tool} failed ({Kind})", tool, kind);
                return ToolResult.Error(kind.ToString().ToUpperInvariant(), SafeMessage(kind), retryable: kind == FailureKind.Transient);    // a short, non-revealing message for the MODEL
            }
        }
    }
}
```

### Repair loops for invalid model output

```csharp
// Structured output can still be wrong (an unknown enum, an impossible value, a tool that doesn't exist). Bounded repair beats blind retry.
public async Task<T> GetValidatedAsync<T>(List<ChatMessage> messages, Func<T, IReadOnlyList<string>> validate, int maxRepairs, CancellationToken ct)
{
    for (int attempt = 0; ; attempt++)
    {
        ChatResponse<T> response = await llm.GetResponseAsync<T>(messages, new ChatOptions { Temperature = 0 }, cancellationToken: ct);
        IReadOnlyList<string> problems = validate(response.Result);

        if (problems.Count == 0) return response.Result;
        if (attempt >= maxRepairs) throw new OutputValidationException(problems);                // give up: escalate or fail — never loop forever

        messages.Add(new(ChatRole.User, "Your previous output was invalid:\n- " + string.Join("\n- ", problems) + "\nReturn a corrected response."));   // FEED THE ERRORS BACK
    }
}
```

### Compensation (the saga pattern) for side effects

```csharp
// When a multi-step task must be all-or-nothing, record an UNDO for each completed side effect; on failure, run them in reverse.
public sealed class Saga
{
    private readonly Stack<(string Name, Func<CancellationToken, Task> Undo)> _compensations = new();

    public async Task<T> StepAsync<T>(string name, Func<Task<T>> action, Func<T, CancellationToken, Task> undo)
    {
        T result = await action();
        _compensations.Push((name, ct => undo(result, ct)));              // register the undo only AFTER the action succeeded
        return result;
    }

    public async Task CompensateAsync(ILogger log, CancellationToken ct)
    {
        while (_compensations.TryPop(out var c))
        {
            try { await c.Undo(ct); log.LogInformation("Compensated {Step}", c.Name); }
            catch (Exception ex) { log.LogError(ex, "COMPENSATION FAILED for {Step} — manual intervention required", c.Name); }      // alert a human: the system is now in a partial state
        }
    }
}

// Some actions can't be undone (an email sent, a payment settled). Prefer REVERSIBLE designs ("create draft", "authorize then capture"), put irreversible steps LAST, and put approvals BEFORE them.
```

### Recovery strategies

```plaintext
RESUME            After a crash: reload state, reconcile in-doubt calls, continue (Section 9).
ROLL BACK         Restore the last good checkpoint and retry a failed step with new information, a different tool, or a stronger model.
REPLAN            The plan assumed something false; generate a new plan around the completed work (Section 2).
ESCALATE          Hand to a stronger model, a specialist agent, or a human with the full context and what was tried (Section 13).
DEGRADE           Return a partial or lower-fidelity result with a clear statement of what's missing, rather than failing completely.
DEAD-LETTER       Park an unrecoverable run with its state in a queue for inspection, with alerts and a runbook; don't retry forever.
KILL SWITCH       Operators can cancel a run or disable a tool/agent immediately (see AI Security, Section 13).
POST-MORTEM       Every unrecovered failure becomes a test case in the evaluation suite (Section 19).
```

---

## 16. Infinite-Loop Prevention

### Agents loop in several different ways

```plaintext
IDENTICAL-CALL LOOP      The same tool with the same arguments, repeatedly (the result never changes and the model doesn't adapt).
OSCILLATION              A -> B -> A -> B: two states or two plans that undo each other.
REFINEMENT LOOP          A critic never accepts, or a generator keeps "improving" past usefulness.
PLANNING LOOP            Replan -> fail -> replan -> fail, each plan equally flawed.
HANDOFF PING-PONG        Two agents keep transferring the conversation (Section 13).
RETRY STORM              Repeated retries of a non-retryable failure.
LOW-PROGRESS DRIFT       Not identical, but endlessly "exploring" with no new information gained — the most common and the hardest to detect.
AGENT-TO-AGENT CHATTER   Multi-agent conversations that never converge.
```

### Layered defenses

```plaintext
1. HARD CAPS (the non-negotiable backstop)   Maximum steps, tool calls, tokens, cost, wall-clock time, retries, handoffs, replans, refinement rounds, delegation depth (Section 17). Every loop in the system has an upper bound in CODE.
2. REPETITION DETECTION                      Fingerprint (tool, normalized arguments); count repeats within a window.
3. CYCLE DETECTION                           Hash the meaningful state (not the whole transcript); a repeat of a previously seen state means a cycle.
4. PROGRESS DETECTION                        Track "new information" per step (new facts, new documents seen, new plan steps completed); N steps without progress triggers intervention.
5. INTERVENTION LADDER                       First nudge ("You've repeated this call; the result won't change — try a different approach or finish"), then restrict (disable the repeated tool or narrow the toolset), then escalate to a human or stop with
                                             a partial result.
6. ACCEPTANCE CRITERIA FOR REFINEMENT        Critic/generator loops need objective stop conditions: a checklist passed, a validator green, or a maximum of 2-3 rounds.
7. NO-OP AWARENESS                           Give the agent an explicit way to say "I can't do this" (a give_up/ask_user tool). Without one, models loop trying.
```

### A loop detector

```csharp
public sealed class LoopDetector(int maxIdenticalCalls = 3, int noProgressWindow = 5)
{
    private readonly Dictionary<string, int> _callCounts = new();
    private readonly HashSet<string> _seenStates = new();
    private int _stepsSinceProgress;

    // 1. Repetition: the same tool + same normalized arguments, again
    public LoopSignal Observe(string tool, IDictionary<string, object?>? arguments)
    {
        string fingerprint = tool + ":" + Canonicalize(arguments);                                        // sort keys, trim strings, normalize numbers
        int count = _callCounts[fingerprint] = _callCounts.GetValueOrDefault(fingerprint) + 1;

        if (count >= maxIdenticalCalls + 2) return LoopSignal.Stop($"'{tool}' called {count} times with the same arguments.");
        if (count >= maxIdenticalCalls)     return LoopSignal.Nudge($"You have called '{tool}' with the same arguments {count} times and the result will not change. Try a different approach, or finish with what you have.");
        return LoopSignal.None;
    }

    // 2. Cycles: the agent's meaningful state returns to a previous value
    public LoopSignal ObserveState(AgentState state)
    {
        string hash = Hash(state.Plan?.CompletedStepIds, state.Facts.Keys.OrderBy(k => k), state.PendingApprovals.Count);     // NOT the transcript — that always changes
        return _seenStates.Add(hash) ? LoopSignal.None : LoopSignal.Stop("The agent returned to a previous state (cycle).");
    }

    // 3. Progress: did this step add anything new?
    public LoopSignal ObserveProgress(bool gainedNewInformation)
    {
        _stepsSinceProgress = gainedNewInformation ? 0 : _stepsSinceProgress + 1;
        return _stepsSinceProgress >= noProgressWindow
            ? LoopSignal.Nudge($"The last {_stepsSinceProgress} steps produced no new information. Summarize what you know and either finish or ask the user for help.")
            : LoopSignal.None;
    }
}

// In the agent loop: apply the intervention ladder
LoopSignal signal = loops.Observe(call.Name, call.Arguments);
if (signal.IsNudge) messages.Add(new ChatMessage(ChatRole.User, signal.Message));                         // 1. nudge once or twice
if (signal.IsStop)  return AgentRunResult.Stopped("loop detected", partial: BuildPartial(messages));       // 3. stop with a partial result (a restricted-toolset retry is the middle rung)
```

### Detecting "no progress" in practice

```plaintext
SIGNALS THAT A STEP ADDED INFORMATION
  - a tool returned results not seen before (new document IDs, new rows, new facts)           - a plan step moved to completed
  - the structured state changed meaningfully                                                 - new evidence was added to the evidence list
SIGNALS OF DRIFT
  - consecutive assistant messages are highly similar (n-gram overlap or embedding similarity)
  - searches with near-identical queries; reading the same documents again
  - growing context with a flat evidence/facts count
Combine cheap signals (fingerprints, counters) as the first line of defense; use similarity checks sparingly.
```

### Principles

```plaintext
EVERY LOOP HAS A BOUND            If you can't state the maximum iterations of a loop, you've written a bug that bills by the token.
DETECT SOON, ESCALATE GRACEFULLY  A nudge early costs a message; a runaway costs real money and trust. Prefer returning a partial result and an explanation over silently continuing or crashing.
DESIGN OUT THE LOOP               Many loops stem from missing exits: no give-up tool, vague success criteria, tool errors without guidance. Fix those before adding detectors.
TEST LOOPS DELIBERATELY           Fault-injection tests: tools that always return the same thing, always fail, or contradict each other; verify the agent stops within bounds (Section 19).
ALERT ON LOOP EVENTS              Loop detections and budget stops are first-class metrics and alerts (see AI Observability, Section 10).
```

---

## 17. Budget and Iteration Limits

### Budgets are how a probabilistic system gets a worst case

Without limits, an agent's cost and duration are unbounded — set by whatever the model decides to do next. **Budgets** convert "it depends" into a guaranteed ceiling. Define them per run, divide them among sub-agents, enforce them in code, and treat hitting a limit as a managed outcome (a graceful partial result), not an exception that corrupts state.

### Dimensions to limit

```plaintext
STEPS / ITERATIONS         Model calls per run (the loop counter).
TOOL CALLS                 Total, and PER TOOL (an expensive or risky tool gets a tighter cap).
TOKENS                     Input and output, cumulative — context re-sending makes long runs grow superlinearly.
COST                       Dollars (computed from tokens and tool pricing) — the limit finance actually cares about.
WALL-CLOCK TIME            Deadlines per run, per step, per tool call.
RETRIES                    Per tool call and per run.
HANDOFFS / REPLANS / REFINEMENT ROUNDS   Each loop type has its own cap.
DELEGATION DEPTH & FAN-OUT Maximum sub-agent depth and the number of concurrent/total workers.
MEMORY WRITES              Writes per run (a runaway agent shouldn't flood long-term memory).
MESSAGES                   Agent-to-agent messages per run and per pair.
PER-TENANT / PER-DAY       Aggregate budgets across runs (see AI Security, Section 11) — protects against many cheap runs adding up.
```

### A hierarchical budget

```csharp
public sealed class BudgetExceededException(string dimension) : Exception($"Budget exceeded: {dimension}") { public string Dimension { get; } = dimension; }

public sealed class RunBudget
{
    private readonly long _maxSteps, _maxTokens, _maxToolCalls, _maxMicroUsd;
    private readonly DateTimeOffset _deadline;
    private readonly RunBudget? _parent;
    private long _steps, _tokens, _toolCalls, _microUsd;

    public RunBudget(long maxSteps, long maxTokens, long maxToolCalls, decimal maxUsd, DateTimeOffset deadline, RunBudget? parent = null)
        => (_maxSteps, _maxTokens, _maxToolCalls, _maxMicroUsd, _deadline, _parent) = (maxSteps, maxTokens, maxToolCalls, (long)(maxUsd * 1_000_000), deadline, parent);

    public bool Exhausted => Volatile.Read(ref _steps) >= _maxSteps || Volatile.Read(ref _tokens) >= _maxTokens || Volatile.Read(ref _toolCalls) >= _maxToolCalls
                          || Volatile.Read(ref _microUsd) >= _maxMicroUsd || DateTimeOffset.UtcNow >= _deadline;

    // Charge consumption. A child's consumption ALWAYS counts against its parent too, so the sum of slices can never exceed the parent's budget.
    public void Charge(long steps = 0, long tokens = 0, long toolCalls = 0, decimal usd = 0)
    {
        _parent?.Charge(steps, tokens, toolCalls, usd);

        if (Interlocked.Add(ref _steps, steps)         > _maxSteps)     throw new BudgetExceededException("steps");
        if (Interlocked.Add(ref _tokens, tokens)        > _maxTokens)    throw new BudgetExceededException("tokens");
        if (Interlocked.Add(ref _toolCalls, toolCalls)  > _maxToolCalls) throw new BudgetExceededException("tool calls");
        if (Interlocked.Add(ref _microUsd, (long)(usd * 1_000_000)) > _maxMicroUsd) throw new BudgetExceededException("cost");
        if (DateTimeOffset.UtcNow >= _deadline)                          throw new BudgetExceededException("time");
    }

    // SOFT limit: warn the agent to wrap up BEFORE the hard limit hits, so it can produce a useful final answer
    public bool NearLimit(double fraction = 0.8) =>
        Volatile.Read(ref _steps) >= _maxSteps * fraction || Volatile.Read(ref _tokens) >= _maxTokens * fraction || Volatile.Read(ref _microUsd) >= _maxMicroUsd * fraction;

    // Carve a slice for a sub-agent out of what REMAINS (not the original total)
    public RunBudget Allocate(double fraction) => new(
        (long)((_maxSteps - Volatile.Read(ref _steps)) * fraction), (long)((_maxTokens - Volatile.Read(ref _tokens)) * fraction),
        (long)((_maxToolCalls - Volatile.Read(ref _toolCalls)) * fraction), (_maxMicroUsd - Volatile.Read(ref _microUsd)) * fraction / 1_000_000m,
        _deadline, parent: this);
}
```

### Using budgets in the loop: soft warnings, hard stops, graceful finish

```csharp
for (int i = 1; ; i++)
{
    // SOFT LIMIT: tell the agent once, while it can still act on it
    if (!wrapUpSent && budget.NearLimit(0.8))
    {
        messages.Add(new ChatMessage(ChatRole.User, "You are near your budget. Stop exploring, and give your best answer from what you have now."));
        wrapUpSent = true;
    }

    try
    {
        budget.Charge(steps: 1);
        ChatResponse response = await llm.GetResponseAsync(messages, options, ct);
        budget.Charge(tokens: response.Usage?.TotalTokenCount ?? 0, usd: pricing.Cost(response.ModelId, response.Usage));
        // ... tool calls charge toolCalls:1 each ...
    }
    catch (BudgetExceededException ex)
    {
        // HARD LIMIT: do not crash. Produce a managed outcome: a partial result from the state gathered so far (made WITHOUT further tool calls).
        string partial = await SummarizeProgressAsync(state, ct);                         // one bounded final call, with its own small reserved allowance
        return AgentRunResult.Partial(partial, reason: $"Stopped: {ex.Dimension} budget reached.");
    }
}
```

### Choosing limits

```plaintext
START FROM EVIDENCE      Run your evaluation set (Section 19), measure the distribution of steps, tokens, and cost for SUCCESSFUL runs, and set the hard cap near the 95th-99th percentile — high enough not to truncate normal runs, low
                         enough to stop runaways. Revisit as the agent changes.
TIER BY TASK             Different task types get different budgets; premium tenants may get larger ones.
RESERVE A WRAP-UP ALLOWANCE  Keep a small reserved slice for the final summary/partial result so hitting the cap still yields something useful.
BUDGET PER TOOL          An expensive search or a risky write tool gets its own cap, independent of the overall budget.
ESTIMATE UP FRONT        For plans, estimate cost before executing (steps x typical tokens) and ask for approval when it exceeds a threshold (Section 10).
EXPOSE IT TO THE MODEL   Tell the agent its limits ("You have 8 steps and 4 searches") — models plan better when they know the constraints.
ACCOUNT FOR EVERYTHING   Include judge calls, memory consolidation, retries, and sub-agents in the cost (see AI Observability, Section 5).
LAYER THEM               Per-step timeout < per-run deadline < per-user daily budget < per-tenant monthly budget < provider-side spend limits — each a backstop for the one inside it.
ALERT AND LEARN          Budget-stop rate is a metric: a rising rate signals regressions (a prompt change, a flaky tool) or abuse.
```

---

## 18. Deterministic Workflows vs. Autonomous Agents

### A spectrum, not a binary

```plaintext
FIXED WORKFLOW          Every step and branch is in code; the model is a function inside individual steps (classify, extract, draft).                    Most predictable.
ROUTED WORKFLOW         A model (or classifier) picks WHICH fixed workflow to run; each workflow is deterministic.
WORKFLOW WITH AGENTIC STEPS   A deterministic skeleton, with bounded agents handling the open-ended steps (a diagnosis step, a research step) — inputs/outputs typed, budgets enforced.
BOUNDED AGENT           The model chooses steps and tools within tight limits, a small toolset, validated outputs, and approval gates.
AUTONOMOUS AGENT        The model plans, chooses tools, delegates, and decides when to stop, with broad tools and long horizons.                           Most flexible, least predictable.
```

### What you give up and gain as you move toward autonomy

| | Deterministic workflow | Autonomous agent |
|---|---|---|
| **Predictability** | High — same inputs, same path | Low — path varies run to run |
| **Testability** | Standard tests plus model evals per step | Statistical evaluation of trajectories; hard to cover |
| **Auditability** | Trivial: the code is the policy | Requires full traces; rationale may be unreliable |
| **Cost and latency** | Known and bounded | Variable; can balloon without budgets |
| **Flexibility** | Handles only anticipated paths | Handles novel, open-ended tasks |
| **Failure modes** | Understood, enumerable | Emergent: loops, drift, wrong tools, compounding errors |
| **Security surface** | Small, static | Larger: tools reachable by injected text |
| **Maintenance** | Add a branch per new case | Prompt/tool tuning; regression risk on every change |
| **Time to first demo** | Slower (design the flow) | Faster (give it tools) |
| **Time to reliable production** | Faster, in well-understood domains | Slower: evaluation, guardrails, monitoring |

### A decision framework

```plaintext
ASK, IN ORDER:
 1. CAN YOU WRITE DOWN THE STEPS?   If the sequence is known (even with branches), use a DETERMINISTIC WORKFLOW with model calls inside steps. This covers most business processes: ticket triage, document processing, approvals, data entry, reporting.
 2. IS ONLY PART OF IT OPEN-ENDED?  Keep the workflow skeleton and put a BOUNDED AGENT in just that step (diagnosis, research, investigation), with typed inputs and outputs, a small toolset, and a budget slice.
 3. DOES THE PATH DEPEND ON WHAT YOU DISCOVER, UNPREDICTABLY?  Genuinely exploratory work (open-ended research, debugging unknown failures, multi-source investigations) justifies an AUTONOMOUS-STYLE agent — still with budgets,
    least privilege, approval gates, and loop detection.
 4. WHAT'S THE COST OF A WRONG ACTION?  The higher it is, the more you want determinism around the action (validation, two-phase tools, approvals) even if reasoning upstream is flexible.
 5. CAN YOU EVALUATE IT?  If you cannot define success and test it, don't grant autonomy; keep a human in the loop.
 6. DOES THE ECONOMICS WORK?  Agents spend many model calls per task. If a workflow or a single call suffices, it's cheaper and faster.
```

### The most reliable pattern: agent proposes, workflow disposes

```csharp
// A refund process with a DETERMINISTIC skeleton and ONE bounded agentic step. The agent investigates and PROPOSES a decision; deterministic code validates and executes it.
public async Task<RefundOutcome> HandleRefundRequestAsync(RefundRequest request, UserContext user, CancellationToken ct)
{
    // 1. DETERMINISTIC: fetch facts and check hard rules in code (no model involved)
    Order? order = await orders.GetForUserAsync(user.Id, request.OrderNumber, ct);
    if (order is null)                    return RefundOutcome.Rejected("Order not found.");
    if (!policy.WithinWindow(order))      return RefundOutcome.Rejected("Outside the refund window.");

    // 2. AGENTIC (bounded): investigate ambiguous cases — damage claims, conflicting delivery records — with read-only tools, a budget slice, and a TYPED output
    InvestigationFindings findings = await investigator.RunAsync(
        new InvestigationInput(order, request.Reason, request.CustomerMessage),
        budget: new RunBudget(maxSteps: 8, maxTokens: 30_000, maxToolCalls: 6, maxUsd: 0.25m, deadline: DateTimeOffset.UtcNow.AddMinutes(2)), ct);

    // 3. DETERMINISTIC: validate the agent's PROPOSAL against policy — the model suggests, the code decides
    RefundDecision decision = policyEngine.Decide(order, request, findings);          // amounts, eligibility, risk tier — all code
    if (decision.RequiresApproval) await approvals.RequestAndWaitAsync(decision, ct);  // durable human approval above thresholds

    // 4. DETERMINISTIC: execute with an idempotency key, then verify the end state
    return await refunds.ExecuteAsync(decision, idempotencyKey: $"refund:{request.RequestId}", ct);
}
```

### Hybrid and progressive-autonomy strategies

```plaintext
START DETERMINISTIC     Build the workflow first. It gives you a baseline, a safety net, and an evaluation harness.
PROMOTE STEPS TO AGENTIC ONLY WHEN EVALS SHOW A GAIN   If a branch is exploding in special cases, or a step needs open-ended exploration, replace that STEP with a bounded agent and measure: success, cost, latency, safety.
ROUTE BY DIFFICULTY      A classifier sends easy, well-understood cases down the cheap deterministic path and only hard cases to an agent.
DEMOTE ON FAILURE        If an agentic step is unreliable, constrain it (smaller toolset, stricter schema, fixed sub-steps) or turn it back into code — autonomy is earned and revocable.
AGENT AS A TOOL FOR THE WORKFLOW   Expose an agent as a workflow step with a typed contract; the workflow remains in charge of ordering, retries, approvals, and budgets.
WORKFLOWS AS TOOLS FOR THE AGENT   Give the agent "macros" — deterministic, tested multi-step procedures (e.g. a refund workflow) as single tools, so it composes reliable building blocks instead of improvising every low-level step.
KEEP DETERMINISM AROUND SIDE EFFECTS   Even in an autonomous design, the final consequential actions (payments, sends, deletes) go through validated, two-phase, approved, idempotent tools.
```

### Quick guidance by scenario

| Scenario | Prefer | Why |
|---|---|---|
| Invoice processing, order routing, standard approvals | Deterministic workflow, model in extraction/classification steps | Known steps, high volume, audit needs |
| Customer support Q&A over documents | Fixed RAG pipeline (see *RAG*) | Predictable, cheap; agent only if multi-step questions are common |
| Support with actions (refunds, changes) | Workflow with a bounded investigation agent + two-phase tools + approvals | Open-ended diagnosis, strictly controlled actions |
| Deep research across many sources | Bounded/autonomous research agent with budgets and citations | Path depends on discoveries |
| Coding assistant on a repo | Agent with sandboxed tools, tests as verification, approvals for merges | Exploratory; verifiable by tests |
| IT/ops incident triage | Workflow skeleton (collect telemetry) + diagnostic agent + runbook macros + human approval for remediation | Mix of fixed collection and open-ended diagnosis |
| Data entry/form filling | Deterministic workflow with model-based extraction and validation | Exactness matters |
| Open-ended personal assistant | Autonomous-style agent with tight permissions and confirmations | Breadth outweighs predictability, but risk is high |

---

## 19. Testing and Observing Advanced Agents

### What's different about testing durable, multi-step agents

Unit tests prove your *code* works; evaluations prove the *behavior* is good (see *AI Agents*, Section 11). Advanced agents add a third need: proving they are **recoverable, bounded, and deterministic where it counts** — which requires fault injection and replay.

### Deterministic tests with a scripted model

```csharp
// A fake chat client that returns a PRE-RECORDED sequence of responses — so the agent loop, state handling, budgets, and recovery can be tested without a real model, deterministically and for free.
public sealed class ScriptedChatClient(Queue<ChatResponse> script) : IChatClient
{
    public List<IReadOnlyList<ChatMessage>> Calls { get; } = [];

    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken ct = default)
    {
        Calls.Add(messages.ToList());
        return Task.FromResult(script.Count > 0 ? script.Dequeue() : throw new InvalidOperationException("Script exhausted: the agent made more model calls than expected."));
    }

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> m, ChatOptions? o = null, CancellationToken ct = default) => throw new NotSupportedException();
    public object? GetService(Type serviceType, object? serviceKey = null) => null;
    public void Dispose() { }
}

[Fact]
public async Task Agent_stops_with_partial_result_when_it_repeats_the_same_tool_call()
{
    // The "model" keeps requesting the same search forever
    var script = new Queue<ChatResponse>(Enumerable.Range(0, 20).Select(_ => ToolCallResponse("search_docs", """{"query":"refund policy"}""")));
    var agent = new ReActAgent(new ScriptedChatClient(script), tools, policy, new RunBudget(20, 100_000, 20, 5m, DateTimeOffset.UtcNow.AddMinutes(1)), new LoopDetector());

    AgentRunResult result = await agent.RunAsync("What is the refund policy?", System, CancellationToken.None);

    Assert.Equal("stopped", result.Status);
    Assert.Contains("loop", result.Reason, StringComparison.OrdinalIgnoreCase);
    Assert.True(result.Steps.Count <= 6);                             // the detector stopped it long before the step cap
}
```

### Crash-and-resume tests

```csharp
[Fact]
public async Task Resumed_run_does_not_repeat_a_completed_side_effect()
{
    var store = new InMemoryEventStore();
    var payments = new RecordingPaymentsApi();                        // counts calls per idempotency key

    // Run 1: a fault injector kills the process right AFTER the refund tool ran but BEFORE its result was recorded (crash window 5)
    var agent1 = BuildAgent(store, payments, crashAfterToolBeforeRecord: true);
    await Assert.ThrowsAsync<SimulatedCrashException>(() => agent1.RunOrResumeAsync("run-1", "Refund order 10482", CancellationToken.None));

    // Run 2: a brand-new agent instance resumes from the persisted events
    var agent2 = BuildAgent(store, payments, crashAfterToolBeforeRecord: false);
    AgentState final = await agent2.RunOrResumeAsync("run-1", goalIfNew: null, CancellationToken.None);

    Assert.Equal(RunStatus.Completed, final.Status);
    Assert.Equal(1, payments.EffectiveRefundsFor("run-1:..."));       // the refund happened EXACTLY ONCE despite the crash and re-execution
}
```

### What to test and how

```plaintext
LOOP AND BUDGET LIMITS     Scripted/fake models that loop, tools that always fail or return the same thing: assert the run stops within bounds with a partial result.
RESUME / IDEMPOTENCY       Kill the run at EVERY point between steps (a fault-injection harness that crashes after the Nth event); resume; assert the final state is correct and side effects occurred exactly once.
REPLAY DETERMINISM         Replay recorded event logs through the reducer: state must match exactly. For durable orchestrations, test that code changes don't break replay of existing histories.
APPROVAL FLOWS             Approved, rejected, edited, expired, and unauthorized-approver cases; proposal-hash mismatch; stale preconditions on resume.
HANDOFFS                   Ping-pong attempts, disallowed targets, briefing completeness, privilege non-transfer.
MULTI-AGENT CONTRACTS      Schema-violating worker results, timeouts, partial failures, reassignment; verify the supervisor degrades to a partial result.
CONCURRENCY                Two workers advancing one run (the optimistic append must reject one); parallel worker budgets never exceeding the parent's.
SCHEMA EVOLUTION           Old-version events and in-flight runs processed by new code (upcasters, pinned versions).
SECURITY                   Injection in tool outputs and inter-agent messages must not escalate tools or permissions (see AI Security, Section 14).
EVALUATION (REAL MODEL)    Task success, trajectory efficiency, tool-selection accuracy, cost/step distributions, repeated runs (pass^k) on a simulated environment (see AI Agents, Section 11).
```

### Observability for advanced agents

```plaintext
TRACES        One trace per run; spans per model call, tool call, worker task, handoff, approval wait, and checkpoint; the run ID and trace context propagate through activities, queues, and agent messages (see AI Observability, Section 6).
              For durable runs, link the spans of resumed segments to the same run.
METRICS       steps per run · tool calls per run (by tool) · tokens and cost per run · run duration · success/partial/failure rates · loop detections · budget stops (by dimension) · retry counts · circuit-breaker opens ·
              approval request→decision latency, approval/rejection/expiry rates · handoff counts and loops · worker failure/timeout rates · resume counts and in-doubt reconciliations · dead-lettered runs
LOGS/EVENTS   The event log IS the audit trail: who/what/when for every decision, tool call, approval, and handoff, with redacted payloads and references to stored artifacts.
DASHBOARDS    Active/waiting/stuck runs · oldest waiting approval · cost per tenant · top failing tools · budget-stop trend · replays/resumes.
ALERTS        Runs exceeding their expected duration · spikes in loop detections or budget stops · dead-letter growth · circuit breakers open · approval backlog · unusual tool usage (security) · cost anomalies.
OPERATOR TOOLS View a run's timeline, inspect state at any event, fork from a checkpoint, cancel/suspend, retry a dead-lettered run, disable a tool/agent (kill switch).
```

---

## 20. Putting It Together

### A reference architecture

```plaintext
 CLIENTS ─> API (authN/Z, rate limits) ─> RUN SERVICE: creates a run (id, pinned versions, budgets) and starts a durable workflow
                                                │
                              ┌─────────────────▼───────────────────────────────────────────────────────────────┐
                              │ DURABLE ORCHESTRATION (Durable Functions / Temporal / event-sourced worker)     │
                              │   workflow skeleton (deterministic)                                             │
                              │     ├─ ROUTER (classify; allow-listed outputs)                                  │
                              │     ├─ deterministic steps (fetch, validate, policy checks)                     │
                              │     ├─ BOUNDED AGENT STEPS (ReAct / plan-execute) ── budget slice, loop detector, tool retrieval │
                              │     ├─ SUPERVISOR ─> WORKERS (typed contracts, parallel, timeouts, recovery)    │
                              │     ├─ HANDOFFS (explicit topology, briefings, loop guards)                     │
                              │     └─ HUMAN APPROVAL (durable wait, quorum, expiry, re-validation on resume)   │
                              │   every model call and tool call = an ACTIVITY (recorded, retried, idempotent)  │
                              └───────┬──────────────────────┬───────────────────────┬──────────────────────────┘
                                      v                      v                       v
                               EVENT STORE / STATE      TOOL LAYER                MODEL GATEWAY
                               (events, snapshots,      (schemas, validation,     (routing, fallbacks,
                                blob refs)               two-phase, policy,        budgets, caching)
                                                         idempotency, circuit
                                                         breakers, egress control)
        Memory service (scoped, write-policy) · Approval service · Message bus (typed envelopes) · MCP/A2A adapters
        OBSERVABILITY: traces · metrics · event-log audit · dashboards · alerts · operator console (inspect, fork, cancel, kill switch)
        SECURITY: identity propagation · least-privilege tools · injection isolation · tenant isolation · secrets in vaults
```

### Design checklist

```plaintext
CONTROL FLOW     [ ] Deterministic skeleton; agents only where the path is genuinely open-ended        [ ] Every loop (steps, retries, replans, refinements, handoffs, depth) has a code-enforced bound
STATE            [ ] Typed, versioned, event-sourced state; large data by reference                      [ ] Single writer per run; idempotent, order-safe events; pinned model/prompt/tool versions
DURABILITY       [ ] Model and tool calls are recorded steps (activities); orchestrator code deterministic [ ] Write-ahead intent + idempotency keys; in-doubt reconciliation; resume tested at every crash point
TOOLS            [ ] Few, focused, well-described tools per agent; retrieval/routing for large toolsets    [ ] Typed schemas, validated in code; helpful error contract; two-phase for consequential actions
APPROVALS        [ ] Risk-tiered policy in code; proposal-bound, authorized, expiring, fail-closed         [ ] Re-validation on resume; edit-and-approve; audit; fatigue monitoring
MULTI-AGENT      [ ] Typed contracts; minimal shared context; one owner per responsibility                 [ ] Supervisor logic is code; workers least-privileged; handoff topology and loop guards
COMMUNICATION    [ ] Envelope with correlation, schema version, expiry, trace context; validated on receipt [ ] Agent identities and authorization; untrusted-content handling; bounded chatter
RECOVERY         [ ] Failure taxonomy; retries only for idempotent transient errors; circuit breakers       [ ] Repair loops (bounded); compensation for atomic tasks; escalation and dead-letter paths
BUDGETS          [ ] Hierarchical budgets (run → worker → tool); soft wrap-up warnings, hard stops          [ ] Graceful partial results; limits derived from evaluation percentiles; per-tenant caps
MEMORY           [ ] Scoped; write policy; provenance; consolidation and expiry; untrusted when recalled    [ ] User control and erasure
TESTING          [ ] Scripted-model tests; fault injection and crash/resume; replay tests                  [ ] Evaluation on a simulated environment with repeated runs; adversarial/security suite
OPERATIONS       [ ] Traces/metrics/event-log audit; operator console; kill switch; runbooks               [ ] Cost attribution and anomaly alerts
```

---

## 21. Common Pitfalls

| Pitfall | Why it hurts | Better approach |
|---|---|---|
| Treating the transcript as the agent's state | Lossy, unbounded, unversioned; can't resume or audit | Typed, versioned, event-sourced state; derive prompts from it (Section 5) |
| Calling the model inside durable orchestrator code | Nondeterminism breaks replay; duplicates or diverges | Wrap every model call in an activity so its result is recorded (Section 8) |
| Using `DateTime.UtcNow`, `Guid.NewGuid()`, or `Task.Delay` in orchestrators | Replay takes a different path | Use the orchestration context's time, GUID, and timer APIs (Section 8) |
| Non-idempotent tools behind retries or resume | Duplicate charges, emails, records | Idempotency keys, write-ahead intent, in-doubt reconciliation (Section 9) |
| Changing orchestrator logic with runs in flight | In-flight instances fail replay | Version orchestrations, additive changes, pin versions, drain old runs (Sections 8-9) |
| Unbounded orchestration history | Slow replay, storage growth | `ContinueAsNew`, small payloads, references to blobs (Section 8) |
| Letting an approval timeout mean "proceed" | Unapproved consequential actions | Fail closed; expire pending approvals as not approved (Section 10) |
| Acting on an old approval without re-checking | World changed since approval | Re-validate preconditions and the proposal hash on resume (Section 10) |
| Approval screens that don't show the real action | Rubber-stamping; unauthorized effects | Show exact arguments/preview and evidence; bind approval to a hash (Section 10) |
| Giving all tools to every agent | Wrong-tool errors, token waste, large blast radius | Per-role toolsets, routing, tool retrieval, permission-aware lists (Section 3) |
| Free-form `query`/`command`/`data` tool parameters | Escape hatch for injection and error | Typed, narrow parameters turned into parameterized operations (Section 4) |
| Trusting tool arguments because a schema exists | Models still produce invalid/hostile values | Validate in code; return helpful errors (Section 4) |
| Single-step "dangerous" tools | A hijacked or confused call executes immediately | Two-phase prepare/commit with bound, single-use tokens (Section 4) |
| Supervisor that is just an LLM in a loop | Top-level failures: loops, no validation, no recovery | Decomposition by model; validation, ordering, budgets, retries, verification in code (Section 12) |
| Forwarding full transcripts between agents | Context bloat, telephone-game errors | Typed task/result contracts with summaries and references (Sections 11-12) |
| Unbounded handoffs | Ping-pong loops, privilege games | Explicit topology, handoff caps, loop detection, escalate to a human (Section 13) |
| Agents trusting each other's messages | Injection hops between agents | Schema-validate every message; treat as untrusted data (Sections 11, 14) |
| Retrying everything | Duplicate effects; storms against failing tools | Classify failures; retry only idempotent transient errors with backoff and circuit breakers (Section 15) |
| No repair strategy for invalid structured output | Brittle runs or infinite re-asking | Bounded repair loops with validation feedback, then escalate (Section 15) |
| Irreversible steps first, approvals after | Can't undo; no gate | Reversible-by-design actions; irreversible steps last, behind approvals; sagas for atomic tasks (Section 15) |
| Only a step cap as loop protection | Slow, expensive stops; misses low-progress drift | Repetition, cycle, and progress detection with an intervention ladder (Section 16) |
| No give-up tool or success criteria | The agent keeps trying | An explicit `give_up`/`ask_user`; objective acceptance criteria for refinement loops (Section 16) |
| Budgets as exceptions that corrupt state | Partial work lost; crashes at the limit | Soft warnings, graceful partial results, reserved wrap-up allowance (Section 17) |
| Sub-agent budgets that can exceed the parent's | Total cost unbounded | Hierarchical budgets where child usage charges the parent (Section 17) |
| Choosing autonomy because it demos well | Unpredictable cost, behavior, and risk in production | Start deterministic; promote steps to agentic only with evidence; keep determinism around side effects (Section 18) |
| Evaluating only happy paths | Loops, crashes, and edge cases ship | Scripted-model tests, crash/resume fault injection, replay tests, adversarial suites (Section 19) |
| No operator controls | Can't stop or inspect a misbehaving run | Console to inspect, fork, cancel, and disable tools/agents; alerts and runbooks (Section 19) |

---

## Quick Reference Table

| Topic | Core idea | Practical takeaway |
|---|---|---|
| ReAct | Thought → Action → Observation loop | Native tool calling by default; step records; don't trust visible reasoning as evidence |
| Planning loops | Plan as validated data, execute, monitor, replan | Validate DAG and tools; parallelize independent steps; verify by real state; bound replans |
| Tool selection | Selection degrades as toolboxes grow | Curate per role, route, retrieve tools, permission-aware lists, disambiguating descriptions |
| Tool schema design | Schema = contract for a probabilistic caller | Few typed params, enums, validation in code, helpful errors, two-phase tools, idempotency keys |
| State management | Typed, versioned, event-sourced state | Messages are derived; single writer; large data by reference; versioned schemas |
| Memory | Short-term compaction; long-term scoped, policy-gated | Write policy, provenance, consolidation, expiry; recalled memory is untrusted data |
| Workflow orchestration | Control flow as an explicit graph | Deterministic skeleton, bounded agent steps, checkpoint after each node |
| Durable execution | Persisted progress, replayable orchestrators | Model/tool calls as activities; deterministic orchestrator; ContinueAsNew; version in-flight runs |
| Checkpointing | Write-ahead intent + idempotency | Reconcile in-doubt calls; at-least-once + idempotent = exactly-once effect; pin versions |
| Human approvals | Durable, authorized, expiring, bound to the proposal | Risk-tiered policy; fail closed; re-validate on resume; edit-and-approve |
| Multi-agent collaboration | Specialization at the cost of complexity | Typed contracts; one owner per task; least privilege; measure against a single-agent baseline |
| Supervisor-worker | Coordinator decomposes, workers execute | Supervisor logic in code; validated results; per-worker budgets; partial results first-class |
| Handoffs | Control transfers to another agent | Explicit topology; compact briefing; privileges don't transfer; loop guards; escalate to humans |
| Communication protocols | MCP (agent↔tools), A2A (agent↔agent), typed envelopes | Schema-validate, authenticate agents, bound chatter, treat messages as untrusted |
| Failure recovery | Classify, then remediate | Retry idempotent transient errors; repair loops; sagas; escalate; dead-letter; kill switch |
| Loop prevention | Many loop shapes | Hard caps + repetition/cycle/progress detection + intervention ladder + give-up exit |
| Budgets and limits | A guaranteed worst case | Hierarchical budgets, soft wrap-up, graceful partials, limits from evaluation percentiles |
| Workflows vs. agents | A spectrum | Start deterministic; agent proposes, workflow disposes; promote/demote autonomy by evidence |
| Testing and observing | Scripted models, fault injection, replay | Crash at every step; exactly-once effects; traces, event-log audit, operator console |

---

## Conclusion

The jump from a working agent to a dependable one is almost entirely an exercise in separating concerns: **let the model decide what to try, and let code own everything that must be reliable.** The ReAct loop gives the model its reasoning-and-acting rhythm; plan-and-execute adds a validated plan as data; tool retrieval and tight, typed, two-phase tool schemas keep its choices accurate and its blast radius small. State lives outside the transcript — typed, versioned, event-sourced — so a run can be audited, forked, and resumed. Workflows make control flow explicit; durable execution (Durable Functions, Temporal, or an event-sourced worker you write yourself) turns every model call and tool call into a recorded step, with write-ahead intent and idempotency keys making crashes, retries, and deployments harmless. Human approvals become durable, authorized, expiring workflow steps bound to the exact proposal, re-validated when the run resumes.

Multi-agent designs earn their complexity only when measurement says a single agent falls short, and then only with typed contracts, one owner per responsibility, least-privilege workers, a supervisor whose logic is code, handoffs with explicit topology and loop guards, and protocols (typed envelopes internally, MCP for tools, A2A across systems) that authenticate agents and treat every message as untrusted. Failures are classified rather than blindly retried; invalid outputs get bounded repair; side effects get compensation; loops are detected by repetition, cycle, and progress signals with an intervention ladder and an explicit way to give up; and hierarchical budgets guarantee a worst case, ending in graceful partial results instead of crashes. Above it all sits the most useful design question: *can you write down the steps?* If yes, build a deterministic workflow and put the model inside it. If only part is open-ended, put a bounded agent in just that step. Reserve autonomy for genuinely exploratory work — and even there, keep determinism around the actions that matter, test with scripted models and crash-and-resume fault injection, and give operators the tools to see, steer, and stop what the agent is doing. Do that, and agents stop being impressive demos that fail unpredictably and become systems you can run, bound, audit, and trust.

---

*Found this useful? Feel free to star the repo, open an issue with corrections, or share the "the deployment restarted mid-refund and the idempotency key was the only thing between us and a double payment" story that made the case for write-ahead intent and replay-safe design click better than any workflow diagram.*
