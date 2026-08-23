import os
import re
import subprocess

with open('architecture.md', 'r') as f:
    content = f.read()

# Find all mermaid blocks
blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)

os.makedirs('architecture_diagrams', exist_ok=True)

for i, block in enumerate(blocks, 1):
    mmd_path = f'architecture_diagrams/diagram_{i}.mmd'
    png_path = f'architecture_diagrams/diagram_{i}.png'
    with open(mmd_path, 'w') as f:
        f.write(block)
    
    print(f"Generating diagram {i}...")
    # Run mmdc
    # we use npx -y @mermaid-js/mermaid-cli
    result = subprocess.run(['npx', '-y', '@mermaid-js/mermaid-cli', '-i', mmd_path, '-o', png_path, '-b', 'transparent'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error generating diagram {i}:")
        print(result.stderr)
    else:
        print(f"Successfully generated {png_path}")

print("Done extracting diagrams.")
