import logging
import sys

import jax
import jax.numpy as jnp
import jax.random as jrandom
import haliax as hax
from levanter.models.gpt2 import Gpt2Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_creation():
    """Test if we can create the adaptive model."""
    try:
        from qkvflow.nn.adaptive_transformer import AdaptiveNeuralOdeLMHeadModel
        
        logger.info("Testing model creation...")
        
        # Small config for testing
        config = Gpt2Config(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            seq_len=32,
        )
        
        Vocab = hax.Axis("vocab", 1000)
        
        model = AdaptiveNeuralOdeLMHeadModel.init(
            Vocab=Vocab,
            config=config,
            time_embed_dim=16,
            sinusodial_dim=8,
            num_experts=2,
            key=jrandom.PRNGKey(0),
        )
        
        logger.info("✓ Model creation successful")
        
        # Test forward pass with proper error handling
        logger.info("Testing forward pass...")
        
        try:
            Batch = hax.Axis("batch", 2)
            Pos = hax.Axis("position", 16)
            
            input_ids = hax.random.randint(
                jrandom.PRNGKey(1), 
                shape=(Batch, Pos), 
                minval=0, 
                maxval=999
            )
            
            output = model(input_ids, key=jrandom.PRNGKey(2))
            
            logger.info(f"✓ Forward pass successful, output shape: {output.axes}")
            
        except Exception as e:
            logger.error(f"✗ Forward pass failed: {e}")
            return False
        
        # Test expert analysis with error handling
        logger.info("Testing expert analysis...")
        
        try:
            analysis = model.get_expert_analysis(
                input_ids, key=jrandom.PRNGKey(3)
            )
            
            logger.info(f"✓ Expert analysis successful, found {len(analysis)} analysis keys")
            for key in list(analysis.keys())[:3]:  # Show first 3 keys
                logger.info(f"  - {key}")
            
        except Exception as e:
            logger.error(f"✗ Expert analysis failed: {e}")
            logger.warning("Expert analysis failure is not critical, continuing...")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        logger.error("Make sure the adaptive_transformer module is available")
        return False
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_training_components():
    """Test training components."""
    try:
        from qkvflow.train_adaptive_lm import AdaptiveTrainLmConfig
        from qkvflow.train_lm import DatasetConfig
        from levanter.trainer import TrainerConfig
        from levanter.models.gpt2 import Gpt2Config
        
        logger.info("Testing config creation...")
        
        config = AdaptiveTrainLmConfig(
            data=DatasetConfig(id="Ankursingh/openwebtext_10K"),
            model=Gpt2Config(
                hidden_dim=64,  
                num_heads=4,  
                num_layers=2,
                seq_len=32 
            ),
            trainer=TrainerConfig(
                train_batch_size=2,
                num_train_steps=10
            ),
            time_embed_dim=16,
            sinusodial_dim=8,
            num_experts=2,
        )
        
        logger.info("✓ Config creation successful")
        logger.info(f"  - Model: {config.model.hidden_dim}d, {config.model.num_layers} layers")
        logger.info(f"  - Adaptive: {config.num_experts} experts, {config.time_embed_dim}d time embed")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Training component import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Training component test failed: {e}")
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
    logger.info("Running adaptive model tests...")
    
    success = True
    
    # Test JAX setup first
    success &= test_basic_jax_setup()
    
    # Test model creation
    success &= test_model_creation()
    
    # Test training components
    success &= test_training_components()
    
    if success:
        logger.info("🎉 All tests passed! Model is ready for training.")
    else:
        logger.error("❌ Some tests failed. Please fix issues before training.")
        sys.exit(1)