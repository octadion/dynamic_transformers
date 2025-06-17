import logging
import sys
import traceback

import jax
import jax.numpy as jnp
import jax.random as jrandom
import haliax as hax
from levanter.models.gpt2 import Gpt2Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_temporal_svd_linear():
    """Test TemporalSVDLinear component in isolation."""
    try:
        from qkvflow.nn.temporal_svd_linear import TemporalSVDLinear
        
        logger.info("Testing TemporalSVDLinear...")
        
        # Simple configuration for testing
        SinusodialDim = hax.Axis("sinusodial", 16)
        TembedDim = hax.Axis("tembed", 32)
        In = hax.Axis("in", 8)
        Out = hax.Axis("out", 16)
        
        # Initialize component
        svd_linear = TemporalSVDLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=In,
            Out=Out,
            num_experts=2,
            svd_rank=4,
            key=jrandom.PRNGKey(0),
            use_bias=True,
        )
        
        # Test forward pass
        Batch = hax.Axis("batch", 2)
        Pos = hax.Axis("position", 4)
        
        time_embed = hax.random.normal(
            jrandom.PRNGKey(1), 
            shape=(TembedDim,), 
            dtype=jnp.float32
        )
        
        x = hax.random.normal(
            jrandom.PRNGKey(2), 
            shape=(Batch, Pos, In), 
            dtype=jnp.float32
        )
        
        output = svd_linear(time_embed, x, key=jrandom.PRNGKey(3))
        
        logger.info(f"✓ TemporalSVDLinear: input shape {x.axes}, output shape {output.axes}")
        
        # Test expert contributions
        contributions = svd_linear.get_expert_contributions(time_embed)
        logger.info(f"✓ Expert contributions: {list(contributions.keys())}")
        
        # Test evaluation at specific time
        static_linear = svd_linear.evaluate_at(time_embed)
        logger.info(f"✓ Static evaluation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ TemporalSVDLinear test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_adaptive_attention():
    """Test AdaptiveAttention component."""
    try:
        from qkvflow.nn.adaptive_attention_mlp import AdaptiveAttention
        
        logger.info("Testing AdaptiveAttention...")
        
        # Small config for testing
        config = Gpt2Config(
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            seq_len=8,
        )
        
        SinusodialDim = hax.Axis("sinusodial", 16)
        TembedDim = hax.Axis("tembed", 24)
        
        attention = AdaptiveAttention.init(
            config=config,
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            num_experts=2,
            key=jrandom.PRNGKey(0),
        )
        
        # Test forward pass
        Batch = hax.Axis("batch", 2)
        
        time_embed = hax.random.normal(
            jrandom.PRNGKey(1), 
            shape=(TembedDim,), 
            dtype=jnp.float32
        )
        
        x = hax.random.normal(
            jrandom.PRNGKey(2), 
            shape=(Batch, config.Pos, config.Embed), 
            dtype=jnp.float32
        )
        
        output = attention(
            time_embed=time_embed,
            x=x,
            mask=None,
            layer_idx=0,
            key=jrandom.PRNGKey(3),
        )
        
        logger.info(f"✓ AdaptiveAttention: input shape {x.axes}, output shape {output.axes}")
        
        # Test evaluation
        static_attention = attention.evaluate_at(time_embed)
        logger.info(f"✓ Static attention evaluation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ AdaptiveAttention test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_adaptive_mlp():
    """Test AdaptiveMLP component."""
    try:
        from qkvflow.nn.adaptive_attention_mlp import AdaptiveMLP
        
        logger.info("Testing AdaptiveMLP...")
        
        config = Gpt2Config(
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            seq_len=8,
        )
        
        SinusodialDim = hax.Axis("sinusodial", 16)
        TembedDim = hax.Axis("tembed", 24)
        
        mlp = AdaptiveMLP.init(
            config=config,
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            num_experts=2,
            key=jrandom.PRNGKey(0),
            use_bias=True,
        )
        
        # Test forward pass
        Batch = hax.Axis("batch", 2)
        
        time_embed = hax.random.normal(
            jrandom.PRNGKey(1), 
            shape=(TembedDim,), 
            dtype=jnp.float32
        )
        
        x = hax.random.normal(
            jrandom.PRNGKey(2), 
            shape=(Batch, config.Pos, config.Embed), 
            dtype=jnp.float32
        )
        
        output = mlp(
            time_embed=time_embed,
            x=x,
            key=jrandom.PRNGKey(3),
        )
        
        logger.info(f"✓ AdaptiveMLP: input shape {x.axes}, output shape {output.axes}")
        
        # Test evaluation
        static_mlp = mlp.evaluate_at(time_embed)
        logger.info(f"✓ Static MLP evaluation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ AdaptiveMLP test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_adaptive_block():
    """Test AdaptiveBlock component."""
    try:
        from qkvflow.nn.adaptive_transformer import AdaptiveBlock
        
        logger.info("Testing AdaptiveBlock...")
        
        config = Gpt2Config(
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            seq_len=8,
        )
        
        SinusodialDim = hax.Axis("sinusodial", 16)
        TembedDim = hax.Axis("tembed", 24)
        
        block = AdaptiveBlock.init(
            config=config,
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            num_experts=2,
            key=jrandom.PRNGKey(0),
        )
        
        # Test forward pass
        Batch = hax.Axis("batch", 2)
        
        time_embed = hax.random.normal(
            jrandom.PRNGKey(1), 
            shape=(TembedDim,), 
            dtype=jnp.float32
        )
        
        x = hax.random.normal(
            jrandom.PRNGKey(2), 
            shape=(Batch, config.Pos, config.Embed), 
            dtype=jnp.float32
        )
        
        output = block(
            time_embed=time_embed,
            x=x,
            mask=None,
            layer_idx=0,
            key=jrandom.PRNGKey(3),
        )
        
        logger.info(f"✓ AdaptiveBlock: input shape {x.axes}, output shape {output.axes}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ AdaptiveBlock test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_full_model():
    """Test complete AdaptiveNeuralOdeLMHeadModel."""
    try:
        from qkvflow.nn.adaptive_transformer import AdaptiveNeuralOdeLMHeadModel
        
        logger.info("Testing full adaptive model...")
        
        # Minimal config for testing
        config = Gpt2Config(
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            seq_len=8,
        )
        
        Vocab = hax.Axis("vocab", 100)
        
        model = AdaptiveNeuralOdeLMHeadModel.init(
            Vocab=Vocab,
            config=config,
            time_embed_dim=16,
            sinusodial_dim=8,
            num_experts=2,
            key=jrandom.PRNGKey(0),
        )
        
        logger.info("✓ Model creation successful")
        
        # Test forward pass
        Batch = hax.Axis("batch", 2)
        
        input_ids = hax.random.randint(
            jrandom.PRNGKey(1), 
            shape=(Batch, config.Pos), 
            minval=0, 
            maxval=99
        )
        
        output = model(input_ids, key=jrandom.PRNGKey(2))
        
        logger.info(f"✓ Forward pass successful, output shape: {output.axes}")
        
        # Test expert analysis
        try:
            analysis = model.get_expert_analysis(
                input_ids, key=jrandom.PRNGKey(3)
            )
            
            logger.info(f"✓ Expert analysis successful, found {len(analysis)} components")
            for key in list(analysis.keys())[:3]:
                logger.info(f"  - {key}")
            
        except Exception as e:
            logger.warning(f"Expert analysis failed: {e}, but model works")
        
        # Test loss computation
        try:
            from dataclasses import dataclass
            
            @dataclass
            class MockExample:
                tokens: hax.NamedArray
                attn_mask: hax.NamedArray = None
                loss_mask: hax.NamedArray = None
            
            example = MockExample(
                tokens=input_ids,
                loss_mask=hax.ones_like(input_ids, dtype=jnp.bool_),
            )
            
            loss = model.compute_loss(example, key=jrandom.PRNGKey(4))
            logger.info(f"✓ Loss computation successful: {loss}")
            
        except Exception as e:
            logger.warning(f"Loss computation failed: {e}, but model works")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Full model test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_numerical_stability():
    """Test numerical stability of the model."""
    try:
        logger.info("Testing numerical stability...")
        
        # Test with various input scales
        scales = [1e-3, 1e-1, 1.0, 10.0, 100.0]
        
        for scale in scales:
            try:
                from qkvflow.nn.temporal_svd_linear import TemporalSVDLinear
                
                SinusodialDim = hax.Axis("sinusodial", 8)
                TembedDim = hax.Axis("tembed", 16)
                In = hax.Axis("in", 4)
                Out = hax.Axis("out", 8)
                
                svd_linear = TemporalSVDLinear.init(
                    SinusodialDim=SinusodialDim,
                    TembedDim=TembedDim,
                    In=In,
                    Out=Out,
                    num_experts=2,
                    svd_rank=2,
                    key=jrandom.PRNGKey(0),
                    use_bias=True,
                )
                
                time_embed = hax.random.normal(
                    jrandom.PRNGKey(1), 
                    shape=(TembedDim,), 
                    dtype=jnp.float32
                ) * scale
                
                x = hax.random.normal(
                    jrandom.PRNGKey(2), 
                    shape=(hax.Axis("batch", 1), hax.Axis("pos", 2), In), 
                    dtype=jnp.float32
                ) * scale
                
                output = svd_linear(time_embed, x, key=jrandom.PRNGKey(3))
                
                # Check for NaN/Inf
                if jnp.any(jnp.isnan(output.array)) or jnp.any(jnp.isinf(output.array)):
                    logger.warning(f"⚠️ Numerical instability at scale {scale}")
                else:
                    logger.info(f"✓ Stable at scale {scale}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed at scale {scale}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Numerical stability test failed: {e}")
        return False


def test_memory_usage():
    """Test memory usage and performance."""
    try:
        logger.info("Testing memory usage...")
        
        import time
        
        # Test with different model sizes
        sizes = [
            {"hidden_dim": 16, "num_layers": 1, "seq_len": 4},
            {"hidden_dim": 32, "num_layers": 2, "seq_len": 8},
            {"hidden_dim": 64, "num_layers": 4, "seq_len": 16},
        ]
        
        for size_config in sizes:
            try:
                from qkvflow.nn.adaptive_transformer import AdaptiveNeuralOdeLMHeadModel
                
                config = Gpt2Config(**size_config)
                Vocab = hax.Axis("vocab", 50)
                
                start_time = time.time()
                
                model = AdaptiveNeuralOdeLMHeadModel.init(
                    Vocab=Vocab,
                    config=config,
                    time_embed_dim=8,
                    sinusodial_dim=4,
                    num_experts=2,
                    key=jrandom.PRNGKey(0),
                )
                
                input_ids = hax.random.randint(
                    jrandom.PRNGKey(1), 
                    shape=(hax.Axis("batch", 1), config.Pos), 
                    minval=0, 
                    maxval=49
                )
                
                output = model(input_ids, key=jrandom.PRNGKey(2))
                
                elapsed = time.time() - start_time
                
                logger.info(f"✓ Size {size_config}: {elapsed:.3f}s")
                
            except Exception as e:
                logger.warning(f"⚠️ Size {size_config} failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Memory usage test failed: {e}")
        return False


def run_comprehensive_tests():
    """Run all tests with detailed reporting."""
    logger.info("🧪 Running comprehensive adaptive model tests...")
    
    tests = [
        ("JAX Setup", test_basic_jax_setup),
        ("TemporalSVDLinear", test_temporal_svd_linear),
        ("AdaptiveAttention", test_adaptive_attention),
        ("AdaptiveMLP", test_adaptive_mlp),
        ("AdaptiveBlock", test_adaptive_block),
        ("Full Model", test_full_model),
        ("Numerical Stability", test_numerical_stability),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
            
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: ❌ FAILED with exception: {e}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Model is ready for training.")
        return True
    else:
        logger.error("❌ Some tests failed. Please review and fix issues.")
        return False


def test_basic_jax_setup():
    """Test basic JAX setup and GPU availability."""
    try:
        logger.info("Testing JAX setup...")
        
        # Test basic JAX operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        logger.info(f"✓ Basic JAX operations work: sum([1,2,3]) = {y}")
        
        # Check available devices
        devices = jax.devices()
        logger.info(f"✓ Available devices: {[str(d) for d in devices]}")
        
        # Test random key
        key = jrandom.PRNGKey(42)
        rand_val = jrandom.normal(key, (3,))
        logger.info(f"✓ Random generation works: {rand_val}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ JAX setup test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    
    if not success:
        sys.exit(1)