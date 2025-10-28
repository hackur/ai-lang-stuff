"""
Circuit Discovery with TransformerLens

This example demonstrates how to discover and analyze computational circuits
in transformer models - the specific pathways of computation that implement
particular algorithms or behaviors.

A circuit is a subgraph of the model's computation that is:
1. Necessary for a specific behavior
2. Sufficient to reproduce that behavior
3. Interpretable in terms of what each component does

This involves:
- Identifying important components via activation patching
- Tracing information flow between components
- Analyzing what each component computes
- Visualizing the discovered circuit

Prerequisites:
    uv add transformer-lens torch numpy matplotlib networkx

Usage:
    python circuit_discovery.py

References:
    - Anthropic's "A Mathematical Framework for Transformer Circuits"
    - "In-context Learning and Induction Heads" (Anthropic)
    - TransformerLens documentation
"""

import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import matplotlib.pyplot as plt
import networkx as nx


@dataclass
class Component:
    """Represents a model component (attention head, MLP, etc.)."""

    layer: int
    component_type: str  # "attn_head", "mlp", "resid"
    head: int = None  # Only for attention heads

    def __str__(self):
        if self.component_type == "attn_head":
            return f"L{self.layer}H{self.head}"
        else:
            return f"L{self.layer}_{self.component_type}"

    def __hash__(self):
        return hash((self.layer, self.component_type, self.head))

    def __eq__(self, other):
        return (
            self.layer == other.layer
            and self.component_type == other.component_type
            and self.head == other.head
        )


class CircuitDiscovery:
    """Tools for discovering computational circuits in transformers."""

    def __init__(self, model: HookedTransformer):
        """
        Initialize circuit discovery.

        Args:
            model: HookedTransformer model to analyze
        """
        self.model = model
        self.device = next(model.parameters()).device

    def compute_direct_effect(
        self, tokens: torch.Tensor, source: Component, target_token_pos: int = -1
    ) -> float:
        """
        Compute direct effect of a component on the output.

        This measures how much the component's output directly contributes
        to the final prediction.

        Args:
            tokens: Input tokens
            source: Component to analyze
            target_token_pos: Position to measure effect on

        Returns:
            Magnitude of direct effect
        """
        # Run model and cache activations
        _, cache = self.model.run_with_cache(tokens)

        # Get component output
        if source.component_type == "attn_head":
            act_name = get_act_name("z", source.layer)
            component_out = cache[act_name][0, :, source.head, :]
        elif source.component_type == "mlp":
            act_name = get_act_name("mlp_out", source.layer)
            component_out = cache[act_name][0, :, :]
        else:
            raise ValueError(f"Unknown component type: {source.component_type}")

        # Project through unembedding to get logit contributions
        # Shape: [seq_len, d_model] @ [d_model, vocab] = [seq_len, vocab]
        logit_contributions = component_out @ self.model.W_U

        # Get contribution at target position
        target_logits = logit_contributions[target_token_pos]

        # Return magnitude (L2 norm) of contribution
        return torch.norm(target_logits).item()

    def compute_connection_strength(
        self, tokens: torch.Tensor, source: Component, target: Component
    ) -> float:
        """
        Compute strength of connection between two components.

        This measures how much information flows from source to target
        through the residual stream.

        Args:
            tokens: Input tokens
            source: Source component
            target: Target component (must be in later layer)

        Returns:
            Connection strength (correlation between outputs)
        """
        if source.layer >= target.layer:
            return 0.0

        # Run model and cache
        _, cache = self.model.run_with_cache(tokens)

        # Get source output
        if source.component_type == "attn_head":
            source_name = get_act_name("result", source.layer)
            source_out = cache[source_name][0, :, source.head, :]
        elif source.component_type == "mlp":
            source_name = get_act_name("mlp_out", source.layer)
            source_out = cache[source_name][0, :, :]
        else:
            raise ValueError(f"Unknown source type: {source.component_type}")

        # Get target input (residual stream before target)
        if target.component_type == "attn_head":
            # Get query input to target head
            target_name = get_act_name("q", target.layer)
            target_in = cache[target_name][0, :, target.head, :]
        elif target.component_type == "mlp":
            # Get input to MLP
            target_name = get_act_name("resid_mid", target.layer)
            target_in = cache[target_name][0, :, :]
        else:
            raise ValueError(f"Unknown target type: {target.component_type}")

        # Compute correlation between source output and target input
        # This approximates information flow
        source_flat = source_out.flatten()
        target_flat = target_in.flatten()

        correlation = torch.corrcoef(torch.stack([source_flat, target_flat]))[0, 1]

        return abs(correlation.item())

    def find_important_components(
        self, tokens: torch.Tensor, threshold: float = 0.1, top_k: int = 20
    ) -> List[Tuple[Component, float]]:
        """
        Identify components with large direct effect on output.

        Args:
            tokens: Input tokens
            threshold: Minimum effect magnitude to include
            top_k: Maximum number of components to return

        Returns:
            List of (component, effect_magnitude) tuples, sorted by magnitude
        """
        print("Finding important components...")
        important = []

        # Check all attention heads
        for layer in range(self.model.cfg.n_layers):
            print(f"  Layer {layer}/{self.model.cfg.n_layers - 1}", end="\r")
            for head in range(self.model.cfg.n_heads):
                component = Component(layer, "attn_head", head)
                effect = self.compute_direct_effect(tokens, component)

                if effect >= threshold:
                    important.append((component, effect))

        # Check all MLPs
        for layer in range(self.model.cfg.n_layers):
            component = Component(layer, "mlp")
            effect = self.compute_direct_effect(tokens, component)

            if effect >= threshold:
                important.append((component, effect))

        print(f"  Layer {self.model.cfg.n_layers - 1}/{self.model.cfg.n_layers - 1} ✓")

        # Sort by effect magnitude
        important.sort(key=lambda x: x[1], reverse=True)

        return important[:top_k]

    def trace_circuit(
        self,
        tokens: torch.Tensor,
        important_components: List[Component],
        connection_threshold: float = 0.3,
    ) -> Dict[Tuple[Component, Component], float]:
        """
        Trace connections between important components to find circuit.

        Args:
            tokens: Input tokens
            important_components: Components to analyze connections between
            connection_threshold: Minimum strength to include connection

        Returns:
            Dictionary mapping (source, target) pairs to connection strengths
        """
        print("\nTracing connections between components...")
        connections = {}

        n_components = len(important_components)
        for i, source in enumerate(important_components):
            print(f"  Component {i + 1}/{n_components}", end="\r")

            for target in important_components:
                # Only check forward connections
                if source.layer < target.layer:
                    strength = self.compute_connection_strength(tokens, source, target)

                    if strength >= connection_threshold:
                        connections[(source, target)] = strength

        print(f"  Component {n_components}/{n_components} ✓")

        return connections

    def visualize_circuit(
        self,
        components: List[Tuple[Component, float]],
        connections: Dict[Tuple[Component, Component], float],
        save_path: str = None,
    ):
        """
        Visualize discovered circuit as a directed graph.

        Args:
            components: List of (component, importance) tuples
            connections: Dictionary of (source, target) -> strength
            save_path: Optional path to save figure
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        for component, importance in components:
            G.add_node(
                str(component),
                importance=importance,
                layer=component.layer,
                component_type=component.component_type,
            )

        # Add edges
        for (source, target), strength in connections.items():
            G.add_edge(str(source), str(target), weight=strength)

        # Layout: organize by layer
        pos = {}
        layer_counts = {}

        # Count components per layer
        for component, _ in components:
            layer = component.layer
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        # Position nodes
        layer_positions = {}
        for component, _ in components:
            layer = component.layer
            if layer not in layer_positions:
                layer_positions[layer] = 0

            x = layer
            y = layer_positions[layer] - (layer_counts[layer] - 1) / 2

            pos[str(component)] = (x, y)
            layer_positions[layer] += 1

        # Draw graph
        plt.figure(figsize=(14, 8))

        # Node colors based on importance
        importances = [data["importance"] for _, data in G.nodes(data=True)]
        max_importance = max(importances) if importances else 1.0

        node_colors = [data["importance"] / max_importance for _, data in G.nodes(data=True)]

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=800, cmap="YlOrRd", vmin=0, vmax=1
        )

        # Draw edges with varying thickness
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1.0

        for (u, v), weight in zip(G.edges(), edge_weights):
            alpha = weight / max_weight
            width = 1 + 3 * (weight / max_weight)

            nx.draw_networkx_edges(
                G,
                pos,
                [(u, v)],
                width=width,
                alpha=alpha,
                edge_color="gray",
                arrows=True,
                arrowsize=15,
                connectionstyle="arc3,rad=0.1",
            )

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

        # Add colorbar for node importance
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=0, vmax=max_importance))
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label="Component Importance")

        plt.title("Discovered Computational Circuit", fontsize=16, fontweight="bold")
        plt.xlabel("Layer", fontsize=12)
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nCircuit visualization saved to: {save_path}")

        plt.show()

    def analyze_component_function(
        self, tokens: torch.Tensor, component: Component, top_k_tokens: int = 10
    ) -> Dict:
        """
        Analyze what a component computes.

        This looks at which tokens the component most strongly promotes
        and suppresses to understand its function.

        Args:
            tokens: Input tokens
            component: Component to analyze
            top_k_tokens: Number of top tokens to show

        Returns:
            Dictionary with promoted and suppressed tokens
        """
        # Run model
        _, cache = self.model.run_with_cache(tokens)

        # Get component output
        if component.component_type == "attn_head":
            act_name = get_act_name("z", component.layer)
            component_out = cache[act_name][0, -1, component.head, :]
        elif component.component_type == "mlp":
            act_name = get_act_name("mlp_out", component.layer)
            component_out = cache[act_name][0, -1, :]
        else:
            raise ValueError(f"Unknown component type: {component.component_type}")

        # Project to vocabulary
        logit_contributions = component_out @ self.model.W_U

        # Find most promoted and suppressed tokens
        top_values, top_indices = torch.topk(logit_contributions, top_k_tokens)
        bottom_values, bottom_indices = torch.topk(logit_contributions, top_k_tokens, largest=False)

        # Decode tokens
        promoted = [
            (self.model.to_string(idx.item()), val.item())
            for idx, val in zip(top_indices, top_values)
        ]
        suppressed = [
            (self.model.to_string(idx.item()), val.item())
            for idx, val in zip(bottom_indices, bottom_values)
        ]

        return {"promoted": promoted, "suppressed": suppressed}


def demonstrate_induction_circuit(model: HookedTransformer):
    """
    Discover the induction circuit in a model.

    The induction circuit implements in-context learning:
    Given: [A][B] ... [A] → Predict [B]

    It involves:
    1. Previous token heads (attend to previous token)
    2. Induction heads (attend to token after previous occurrence)
    """
    print("\n" + "=" * 70)
    print("DISCOVERING INDUCTION CIRCUIT")
    print("=" * 70)
    print("\nTask: Complete repeated sequence [A][B]...[A] → [B]")
    print("This is the basic mechanism for in-context learning!")

    discovery = CircuitDiscovery(model)

    # Create repeated sequence
    prompt = "The cat sat on the mat. The dog sat on the rug. The cat sat on the"
    tokens = model.to_tokens(prompt)

    print(f"\nPrompt: {prompt}")
    print("Expected completion: 'mat' (repeating the pattern)")

    # Find important components
    print("\n1. Identifying important components...")
    important = discovery.find_important_components(tokens, threshold=0.5, top_k=15)

    print(f"\nFound {len(important)} important components:")
    for comp, importance in important[:10]:
        print(f"  {comp}: {importance:.3f}")

    # Trace connections
    components_only = [comp for comp, _ in important]
    connections = discovery.trace_circuit(tokens, components_only, connection_threshold=0.3)

    print(f"\nFound {len(connections)} significant connections")

    # Visualize
    print("\n2. Visualizing circuit...")
    discovery.visualize_circuit(important, connections, save_path="induction_circuit.png")

    # Analyze key components
    print("\n3. Analyzing component functions...")
    print("=" * 70)

    for comp, importance in important[:5]:
        print(f"\n{comp} (importance: {importance:.3f})")
        print("-" * 40)

        analysis = discovery.analyze_component_function(tokens, comp, top_k_tokens=5)

        print("Most promoted tokens:")
        for token, value in analysis["promoted"]:
            print(f"  '{token}': {value:+.3f}")

    return important, connections


def demonstrate_factual_circuit(model: HookedTransformer):
    """
    Discover circuit for factual recall.

    Example: "The Eiffel Tower is in" → "Paris"

    This typically involves:
    1. Early layers: recognize entity
    2. Middle layers: retrieve associated fact
    3. Late layers: format output
    """
    print("\n\n" + "=" * 70)
    print("DISCOVERING FACTUAL RECALL CIRCUIT")
    print("=" * 70)
    print("\nTask: Recall factual knowledge")

    discovery = CircuitDiscovery(model)

    prompt = "The Eiffel Tower is located in"
    tokens = model.to_tokens(prompt)

    print(f"Prompt: {prompt}")
    print("Expected completion: 'Paris'")

    # Find important components
    print("\n1. Identifying important components...")
    important = discovery.find_important_components(tokens, threshold=0.3, top_k=15)

    print(f"\nFound {len(important)} important components:")
    for comp, importance in important[:10]:
        print(f"  {comp}: {importance:.3f}")

    # Trace connections
    components_only = [comp for comp, _ in important]
    connections = discovery.trace_circuit(tokens, components_only, connection_threshold=0.25)

    print(f"\nFound {len(connections)} significant connections")

    # Visualize
    print("\n2. Visualizing circuit...")
    discovery.visualize_circuit(important, connections, save_path="factual_circuit.png")

    # Analyze components
    print("\n3. Analyzing component functions...")
    print("=" * 70)

    for comp, importance in important[:3]:
        print(f"\n{comp} (importance: {importance:.3f})")
        print("-" * 40)

        analysis = discovery.analyze_component_function(tokens, comp, top_k_tokens=5)

        print("Most promoted tokens:")
        for token, value in analysis["promoted"]:
            print(f"  '{token}': {value:+.3f}")


def main():
    """Run circuit discovery demonstrations."""

    # Load model
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Loading model on {device}...")
    model = HookedTransformer.from_pretrained(
        "gpt2-small", center_unembed=True, center_writing_weights=True, fold_ln=True, device=device
    )

    # Discover induction circuit
    demonstrate_induction_circuit(model)

    # Discover factual recall circuit
    demonstrate_factual_circuit(model)

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY: UNDERSTANDING CIRCUITS")
    print("=" * 70)
    print(
        """
Key Concepts:

1. WHAT IS A CIRCUIT?
   - A subgraph of model computation
   - Implements a specific algorithm/behavior
   - Necessary and sufficient for that behavior
   - Interpretable at the component level

2. CIRCUIT DISCOVERY PROCESS:
   a) Identify important components (high direct effect)
   b) Trace information flow between components
   c) Analyze what each component computes
   d) Validate the circuit performs the function

3. COMMON CIRCUITS IN TRANSFORMERS:
   - Induction heads: [A][B]...[A] → [B]
   - Previous token heads: attend to previous token
   - Duplicate token heads: identify repeated tokens
   - Factual recall: entity → attribute retrieval

4. WHY CIRCUITS MATTER:
   - Interpretability: understand how models work
   - Debugging: fix specific misbehaviors
   - Safety: identify concerning capabilities
   - Model editing: modify specific functions

5. CIRCUIT PROPERTIES:
   - Composition: circuits build on each other
   - Universality: similar circuits across models
   - Modularity: circuits can be isolated
   - Emergence: complex from simple components

6. NEXT STEPS:
   - Study circuits for specific tasks
   - Analyze circuit composition (how they combine)
   - Test circuit sufficiency (can it work alone?)
   - Investigate failure modes

7. FURTHER READING:
   - "A Mathematical Framework for Transformer Circuits" (Anthropic)
   - "In-context Learning and Induction Heads" (Anthropic)
   - "Interpretability in the Wild" (Redwood Research)
   - TransformerLens documentation

The field of mechanistic interpretability is rapidly evolving.
Circuits are a powerful tool for understanding neural networks at
a mechanistic level, moving beyond just correlations to causal
understanding of how models implement specific computations.
    """
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
