"""
Complete integration of SVD Neural ODE with Transformer² concepts
Combines all components into a working model
"""

import dataclasses
from typing import Dict, Optional, Union, Any
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split

from levanter.models.gpt2 import Gpt2Config, Gpt2Embeddings
from qkvflow.nn.time_embed import SinusoidalPosEmb


class SVDNeuralODELMHeadModel(eqx.Module):
    """
    Complete SVD Neural ODE Language Model
    Integrates SVD decomposition, Neural ODE, and adaptive policies
    """
    
    transformer: SVDNeuralODETransformer
    embeddings: Gpt2Embeddings
    policy: Optional[SVDODEPolicy] = None
    
    # Training configuration
    config: Dict[str, Any] = eqx.field(static=True)
    use_adaptive_mixing: bool = eqx.field(static=True)
    
    @property
    def model_config(self):
        return self.transformer.config
    
    @property
    def Vocab(self) -> hax.Axis:
        return self.embeddings.Vocab
    
    @property
    def Pos(self) -> hax.Axis:
        return self.model_config.Pos
    
    @classmethod
    def init(
        cls,
        Vocab: hax.Axis,
        config: Gpt2Config,
        # SVD-specific parameters
        rank: int = 64,
        num_experts: int = 4,
        # Neural ODE parameters
        time_embed_dim: int = 128,
        sinusodial_dim: int = 32,
        # Adaptive mixing parameters
        use_adaptive_mixing: bool = True,
        # Training configuration
        training_config: Optional[Dict[str, Any]] = None,
        *,
        key,
    ) -> "SVDNeuralODELMHeadModel":
        
        k_transformer, k_embeddings, k_policy = jrandom.split(key, 3)
        
        # Initialize SVD Neural ODE Transformer
        transformer = SVDNeuralODETransformer.init(
            config=config,
            time_embed_dim=time_embed_dim,
            sinusodial_dim=sinusodial_dim,
            rank=rank,
            num_experts=num_experts,
            adaptive_mixing=use_adaptive_mixing,
            key=k_transformer,
        )
        
        # Initialize embeddings
        embeddings = Gpt2Embeddings.init(Vocab, config, key=k_embeddings)
        
        # Initialize policy if using adaptive mixing
        policy = None
        if use_adaptive_mixing:
            policy = SVDODEPolicy.init(
                config=config,
                time_embed_dim=time_embed_dim,
                num_experts=num_experts,
                num_layers=config.num_layers,
                key=k_policy,
            )
        
        # Default training configuration
        if training_config is None:
            training_config = {
                "optimizer_type": "reinforce",
                "learning_rate": 1e-4,
                "entropy_coeff": 0.01,
                "value_loss_coeff": 0.5,
            }
        
        return SVDNeuralODELMHeadModel(
            transformer=transformer,
            embeddings=embeddings,
            policy=policy,
            config=training_config,
            use_adaptive_mixing=use_adaptive_mixing,
        )
    
    def __call__(
        self, 
        input_ids: NamedArray, 
        attn_mask=None, 
        use_policy: bool = True,
        *, 
        key=None
    ) -> NamedArray:
        """
        Forward pass with optional policy control
        """
        k_embed, k_transformer = maybe_rng_split(key, 2)
        
        # Embed input tokens
        x = self.embeddings.embed(input_ids, key=k_embed)
        
        # Apply transformer with optional policy control
        if use_policy and self.policy is not None and self.use_adaptive_mixing:
            # Use policy-controlled forward pass
            x = self._policy_controlled_forward(x, attn_mask, key=k_transformer)
        else:
            # Standard forward pass
            x = self.transformer(x, attn_mask, key=k_transformer)
        
        # Generate logits
        lm_logits = self.embeddings.unembed(x)
        
        return lm_logits
    
    def _policy_controlled_forward(self, x: NamedArray, attn_mask, *, key=None):
        """
        Forward pass controlled by policy decisions
        """
        # Get time embeddings
        t = (hax.arange(self.transformer.config.Layers, dtype=x.dtype) + 1) * self.transformer.dt
        time_embed = self.transformer.time_embedding(t)
        
        if key is not None:
            keys = maybe_rng_split(key, self.transformer.config.num_layers)
        else:
            keys = None
        
        # Process each layer with policy control
        for layer_idx in range(self.transformer.config.num_layers):
            layer_key = keys[layer_idx] if keys is not None else None
            t_embed = time_embed.take("layers", layer_idx)
            
            # Get policy actions for this layer
            policy_actions = self.policy.get_action(t_embed, key=layer_key)
            
            # Apply transformer block with policy-controlled expert mixing
            # This would require modifying the block to accept policy actions
            output = self.transformer.block(
                t_embed, x, attn_mask, layer_idx, key=layer_key
            )
            
            # Apply adaptive time step based on policy
            dt = self.transformer.dt
            if "time_weights" in policy_actions:
                time_weight = policy_actions["time_weights"].take("layers", layer_idx)
                dt = dt * (1.0 + 0.1 * time_weight)  # Small modulation
            
            x = x + output * dt
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        return x
    
    def compute_loss(
        self,
        example,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Compute language modeling loss
        """
        logits = self(example.tokens, example.attn_mask, key=key)
        targets = hax.roll(example.tokens, -1, axis=self.Pos.name)
        target_y = hax.nn.one_hot(targets, self.Vocab, dtype=logits.dtype)
        
        return hnn.cross_entropy_loss(
            logits,
            self.Vocab,
            target_y,
            reduction,
            reduction_axis=reduction_axis,
            where=example.loss_mask,
        )
    
    def compute_policy_loss(
        self,
        example,
        *,
        key=None,
    ) -> Dict[str, NamedArray]:
        """
        Compute policy-related losses for training
        """
        if not self.use_adaptive_mixing or self.policy is None:
            return {"policy_loss": jnp.array(0.0)}
        
        # Get time embeddings
        t = (hax.arange(self.transformer.config.Layers, dtype=example.tokens.dtype) + 1) * self.transformer.dt
        time_embed = self.transformer.time_embedding(t)
        
        # Forward pass with policy tracking
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_bonus = 0.0
        
        # Get model performance as reward
        with jax.no_grad():
            lm_loss = self.compute_loss(example, key=key)
            reward = -lm_loss  # Negative loss as reward
        
        # Compute policy losses across time steps
        for layer_idx in range(self.transformer.config.num_layers):
            t_embed = time_embed.take("layers", layer_idx)
            
            # Get policy actions and values
            actions = self.policy.get_action(t_embed)
            value = self.policy.get_value(t_embed)
            
            # Compute advantage (simplified)
            advantage = reward - value.squeeze()
            
            # Policy gradient loss
            log_probs = self.policy.get_log_probs(t_embed, actions)
            policy_loss = sum(-log_prob * advantage.detach() for log_prob in log_probs.values())
            
            # Value function loss
            value_loss = (value.squeeze() - reward) ** 2
            
            # Entropy bonus
            entropy_bonus = 0.0
            for action_name, action_probs in actions.items():
                if action_name != "time_weights":
                    entropy = -(action_probs * hax.log(action_probs + 1e-8)).sum()
                    entropy_bonus += entropy
            
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy_bonus += entropy_bonus
        
        # Average across layers
        num_layers = self.transformer.config.num_layers
        total_policy_loss /= num_layers
        total_value_loss /= num_layers
        total_entropy_bonus /= num_layers
        
        # Combine losses
        combined_loss = (
            total_policy_loss +
            self.config["value_loss_coeff"] * total_value_loss -
            self.config["entropy_coeff"] * total_entropy_bonus
        )
        
        return {
            "policy_loss": combined_loss,
            "lm_loss": lm_loss,
            "value_loss": total_value_loss,
            "entropy": total_entropy_bonus,
            "reward": reward,
        }
    
    def analyze_expert_usage(self, example, *, key=None) -> Dict[str, Any]:
        """
        Analyze expert usage patterns throughout the model
        """
        if not self.use_adaptive_mixing or self.policy is None:
            return {"message": "No adaptive mixing enabled"}
        
        # Get time embeddings
        t = (hax.arange(self.transformer.config.Layers, dtype=example.tokens.dtype) + 1) * self.transformer.dt
        time_embed = self.transformer.time_embedding(t)
        
        expert_usage = {
            "attention_experts": [],
            "mlp_experts": [],
            "time_modulations": [],
            "rank_selections": [],
        }
        
        for layer_idx in range(self.transformer.config.num_layers):
            t_embed = time_embed.take("layers", layer_idx)
            actions = self.policy.get_action(t_embed)
            
            expert_usage["attention_experts"].append(actions["attention_experts"].array)
            expert_usage["mlp_experts"].append(actions["mlp_experts"].array)
            expert_usage["time_modulations"].append(actions["time_weights"].array)
            expert_usage["rank_selections"].append(actions["rank_selection"].array)
        
        # Convert to numpy for analysis
        for key, values in expert_usage.items():
            expert_usage[key] = jnp.stack(values)
        
        return expert_usage
    
    def get_effective_parameters(self, layer_idx: int = 0) -> Dict[str, NamedArray]:
        """
        Get effective parameters at a specific layer for analysis
        """
        # Get time embedding for the specified layer
        t = (layer_idx + 1) * self.transformer.dt
        time_embed = self.transformer.time_embedding(t)
        
        # Get effective weights from SVD components
        attn_weight = self.transformer.block.attn.c_attn.get_effective_weight(time_embed)
        mlp_weight_fc = self.transformer.block.mlp.c_fc.get_effective_weight(time_embed)
        mlp_weight_proj = self.transformer.block.mlp.c_proj.get_effective_weight(time_embed)
        
        return {
            "attention_qkv": attn_weight,
            "mlp_fc": mlp_weight_fc,
            "mlp_proj": mlp_weight_proj,
        }
    
    @property
    def vocab_size(self) -> int:
        return self.Vocab.size
    
    def resize_vocab(self, new_size: int, key=None) -> "SVDNeuralODELMHeadModel":
        """Resize vocabulary"""
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)


# Example usage and training setup
def create_svd_ode_model_example():
    """
    Example of how to create and use the SVD Neural ODE model
    """
    
    # Model configuration
    config = Gpt2Config(
        seq_len=512,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        vocab_size=50257,
    )
    
    Vocab = hax.Axis("vocab", config.vocab_size)
    
    # Create model with SVD and adaptive mixing
    model = SVDNeuralODELMHeadModel.init(
        Vocab=Vocab,
        config=config,
        rank=64,  # SVD rank
        num_experts=4,  # Number of expert decompositions
        time_embed_dim=128,  # Time embedding dimension
        sinusodial_dim=32,  # Sinusoidal embedding dimension
        use_adaptive_mixing=True,  # Enable policy-based adaptation
        key=jrandom.PRNGKey(42),
    )
    
    return model, config


def train_svd_ode_model(model, train_dataloader, num_steps: int = 1000):
    """
    Example training loop for SVD Neural ODE model
    """
    
    # Initialize optimizer (this would use your preferred JAX optimizer)
    optimizer = SVDODEReinforce(
        learning_rate=1e-4,
        entropy_coeff=0.01,
        value_loss_coeff=0.5,
    )
    
    # Training loop
    for step in range(num_steps):
        # Get batch
        batch = next(train_dataloader)
        
        # Compute losses
        loss_info = model.compute_policy_loss(batch, key=jrandom.PRNGKey(step))
        
        # Optimization step (simplified)
        optimizer.step_optimization(
            model=model,
            policy=model.policy,
            batch_data=batch,
            eval_fn=lambda x: model.compute_loss(x).item(),
        )
        
        # Logging
        if step % 100 == 0:
            print(f"Step {step}:")
            print(f"  LM Loss: {loss_info['lm_loss']:.4f}")
            print(f"  Policy Loss: {loss_info['policy_loss']:.4f}")
            print(f"  Reward: {loss_info['reward']:.4f}")
            
            # Analyze expert usage
            expert_analysis = model.analyze_expert_usage(batch)
            print(f"  Expert diversity: {expert_analysis}")


def analyze_model_adaptation(model, test_examples):
    """
    Analyze how the model adapts its behavior
    """
    
    results = []
    
    for i, example in enumerate(test_examples):
        # Get expert usage for this example
        expert_usage = model.analyze_expert_usage(example)
        
        # Get effective parameters at different layers
        param_analysis = {}
        for layer in [0, model.transformer.config.num_layers // 2, model.transformer.config.num_layers - 1]:
            param_analysis[f"layer_{layer}"] = model.get_effective_parameters(layer)
        
        # Compute performance
        loss = model.compute_loss(example)
        
        results.append({
            "example_id": i,
            "loss": loss.item(),
            "expert_usage": expert_usage,
            "parameter_analysis": param_analysis,
        })
    
    return results


if __name__ == "__main__":
    # Example usage
    model, config = create_svd_ode_model_example()
    
    print("Created SVD Neural ODE model with:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.num_layers}") 
    print(f"  SVD rank: 64")
    print(f"  Num experts: 4")
    print(f"  Adaptive mixing: {model.use_adaptive_mixing}")
    
    # Example forward pass
    Pos = config.Pos
    batch_size = 2
    seq_len = 128
    
    Batch = hax.Axis("batch", batch_size)
    input_ids = hax.random.randint(
        jrandom.PRNGKey(0), 
        (Batch, Pos.resize(seq_len)), 
        minval=0, 
        maxval=config.vocab_size
    )
    
    # Forward pass
    logits = model(input_ids, key=jrandom.PRNGKey(1))
    print(f"Output shape: {logits.shape}")
    
    # Analyze expert usage
    example = type('Example', (), {
        'tokens': input_ids,
        'attn_mask': None,
        'loss_mask': hax.ones_like(input_ids, dtype=bool)
    })()
    
    expert_analysis = model.analyze_expert_usage(example)
    print("Expert usage analysis:", {k: v.shape for k, v in expert_analysis.items()})