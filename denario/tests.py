# from SR_module.PySR_module import PySRModule
# import numpy as np

# SymbolicRegression = PySRModule()

# result = SymbolicRegression.fit(X=np.random.random((100, 2)), y=np.random.random(100), variable_names=["x1", "x2"])
# print("Discovered Equation:", result.equation)
# print("R² Score:", result.r2)

import os
import sys

def test_ollama_available():
    """Check if Ollama is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running with {len(models)} models available:")
            for m in models[:5]:  # Show first 5
                print(f"   - {m['name']}")
            if len(models) > 5:
                print(f"   ... and {len(models) - 5} more")
            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running. Start it with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


def test_langchain_ollama():
    """Test langchain-ollama integration."""
    try:
        from langchain_ollama import ChatOllama
        print("✅ langchain-ollama is installed")
        
        # Try to create a model
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.7,
            base_url="http://localhost:11434",
            num_ctx=8192,
        )
        print("✅ ChatOllama initialized successfully")
        
        # Try a simple call
        print("   Testing simple generation...")
        response = llm.invoke("Say 'Hello from Ollama!' and nothing else.")
        print(f"   Response: {response.content[:100]}...")
        print("✅ Ollama LLM call successful")
        return True
        
    except ImportError:
        print("❌ langchain-ollama not installed. Run: pip install langchain-ollama")
        return False
    except Exception as e:
        print(f"❌ Error with ChatOllama: {e}")
        return False


def test_denario_import():
    """Test that Denario can be imported with Ollama models."""
    try:
        from denario import Denario, models
        print("✅ Denario imported successfully")
        
        # Check if Ollama models are available
        ollama_models = [k for k in models.keys() if 'ollama' in k]
        if ollama_models:
            print(f"✅ Ollama models available: {ollama_models}")
        else:
            print("⚠️  No Ollama models found in models dict (patch may not be applied)")
        
        return True
    except ImportError as e:
        print(f"❌ Cannot import Denario: {e}")
        return False


def test_denario_idea_generation():
    """Test idea generation with Ollama."""
    try:
        from denario import Denario, models
        import tempfile
        
        # Create temporary project directory
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\n📁 Creating test project in {tmpdir}")
            
            den = Denario(project_dir=tmpdir)
            
            # Set a simple data description
            den.set_data_description("""
# Test Data Description

We have a simple physics dataset with the following columns:
- mass: object mass in kg
- velocity: velocity in m/s
- energy: kinetic energy in Joules

The goal is to discover the relationship between these variables.
            """)
            
            print("✅ Data description set")
            
            # Try idea generation with Ollama
            print("\n🧠 Generating idea with Ollama (this may take a minute)...")
            den.get_idea(mode="fast", llm="ollama/llama3.2")
            
            print(f"✅ Idea generated successfully!")
            print(f"\n--- Generated Idea ---")
            print(den.research.idea[:500] + "..." if len(den.research.idea) > 500 else den.research.idea)
            
            return True
            
    except Exception as e:
        print(f"❌ Error during idea generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("DENARIO OLLAMA INTEGRATION TEST")
    print("=" * 60)
    
    results = []
    
    # # Test 1: Ollama availability
    # print("\n[1/4] Checking Ollama availability...")
    # results.append(("Ollama Available", test_ollama_available()))
    
    # # Test 2: langchain-ollama
    # print("\n[2/4] Testing langchain-ollama...")
    # results.append(("langchain-ollama", test_langchain_ollama()))
    
    # Test 3: Denario import
    print("\n[3/4] Testing Denario import...")
    results.append(("Denario Import", test_denario_import()))
    
    # Test 4: Full idea generation (only if previous tests passed)
    if all(r[1] for r in results):
        print("\n[4/4] Testing full idea generation...")
        results.append(("Idea Generation", test_denario_idea_generation()))
    else:
        print("\n[4/4] Skipping idea generation (previous tests failed)")
        results.append(("Idea Generation", None))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        if passed is None:
            status = "⏭️  SKIPPED"
        elif passed:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {name}: {status}")
    
    # Overall result
    passed_count = sum(1 for _, p in results if p is True)
    total_count = sum(1 for _, p in results if p is not None)
    print(f"\nPassed: {passed_count}/{total_count}")
    
    return all(p for _, p in results if p is not None)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
