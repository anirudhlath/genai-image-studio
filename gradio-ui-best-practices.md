# Gradio UI Best Practices

## Component Organization

### **Global vs Tab-Specific Controls**
```python
# ✅ Best Practice: Global settings outside tabs
with gr.Blocks() as demo:
    gr.Markdown("# App Title")

    # Global configuration section
    gr.Markdown("### 🎛️ Global Settings")
    with gr.Row():
        global_model = gr.Dropdown(...)
        global_precision = gr.Dropdown(...)

    # Function-specific tabs
    with gr.Tab("Train"):
        train_specific_controls()
    with gr.Tab("Generate"):
        generate_specific_controls()
```

### **Advanced Settings Pattern**
```python
# ✅ Clean advanced settings grouping
with gr.Column(scale=3):
    with gr.Group():
        gr.Markdown("**Advanced Settings**")
        with gr.Row():
            advanced_setting1 = gr.Dropdown(...)
            advanced_setting2 = gr.Dropdown(...)
            advanced_setting3 = gr.Checkbox(...)
```

## Layout Scaling

### **Balanced Column Layouts**
```python
# ✅ Proportional scaling
with gr.Row():
    with gr.Column(scale=2):  # Primary setting
        main_control = gr.Dropdown(...)
    with gr.Column(scale=2):  # Equal importance
        alt_control = gr.Textbox(...)
    with gr.Column(scale=3):  # More complex content
        grouped_controls()

# ❌ Avoid unbalanced layouts
with gr.Row():
    tiny_control = gr.Checkbox(...)  # No scale - will be cramped
    huge_control = gr.Textbox(...)   # Takes all remaining space
```

## Event Handler Patterns

### **Accessing Global Settings in Local Functions**
```python
def process_with_global_settings(local_param1, local_param2):
    # Access global component values inside event handlers
    model = global_model.value
    precision = global_precision.value

    return actual_function(model, precision, local_param1, local_param2)

# Wire up event handlers with minimal parameters
button.click(
    process_with_global_settings,
    [local_input1, local_input2],  # Only pass tab-specific inputs
    [output_component]
)
```

### **Dynamic Updates with Global Dependencies**
```python
# Update components based on global settings changes
global_precision.change(
    lambda precision, steps, batch_size: estimate_time(steps, batch_size, precision),
    [global_precision, steps_slider, batch_slider],
    [time_estimate_output]
)
```

## Component Configuration

### **Effective Use of Info Text**
```python
# ✅ Concise, actionable info
model_dropdown = gr.Dropdown(
    choices=models,
    label="Base Model",
    info="Select pre-configured model or use custom ID"  # Clear, brief
)

# ❌ Avoid verbose descriptions
model_dropdown = gr.Dropdown(
    choices=models,
    label="Base Model",
    info="This dropdown allows you to select from a list of pre-configured models..."  # Too wordy
)
```

### **Smart Placeholder Usage**
```python
# ✅ Helpful examples
custom_model = gr.Textbox(
    label="Custom Model ID (Optional)",
    placeholder="e.g. runwayml/stable-diffusion-v1-5",  # Shows expected format
    info="Override base model with HuggingFace model ID"
)
```

### **Progressive Disclosure**
```python
# ✅ Hide complexity behind accordions
with gr.Accordion("Advanced Settings", open=False):
    technical_setting1 = gr.Slider(...)
    technical_setting2 = gr.Dropdown(...)
    experimental_setting = gr.Checkbox(...)
```

## Common Gradio Gotchas

### **Component Reference Issues**
```python
# ❌ Components defined in wrong scope
def create_ui():
    with gr.Row():
        input1 = gr.Textbox()

    # This won't work - input1 not accessible here
    def process():
        return input1.value  # Error!

    button.click(process, [], [])

# ✅ Proper scoping
def create_ui():
    with gr.Row():
        input1 = gr.Textbox()

    def process(input_val):  # Pass as parameter
        return f"Processed: {input_val}"

    button.click(process, [input1], [output])
```

### **Import Organization**
```python
# ✅ Import constants where needed
def create_tab():
    # Import at function level to avoid circular imports
    from ..config.constants import PIPELINE_MAPPING, PRECISION_OPTIONS

    pipeline_dropdown = gr.Dropdown(list(PIPELINE_MAPPING.keys()))
```

## Visual Polish

### **Section Headers**
```python
# ✅ Use emoji and formatting for visual hierarchy
gr.Markdown("### 🎛️ Model Configuration")
gr.Markdown("**Advanced Settings**")  # Bold for sub-sections

# ❌ Plain text headers blend together
gr.Markdown("Model Configuration")
gr.Markdown("Advanced Settings")
```

### **Consistent Spacing**
```python
# ✅ Logical grouping with gr.Group()
with gr.Group():  # Adds subtle visual grouping
    related_control1 = gr.Dropdown(...)
    related_control2 = gr.Slider(...)

# Use gr.Row() and gr.Column() for layout, gr.Group() for visual grouping
```

## Performance Considerations

### **Minimize Re-renders**
- Group related updates together
- Use appropriate event triggers (change vs click)
- Avoid updating components unnecessarily

### **Efficient Model Loading**
```python
# ✅ Load models once, reference globally
models = fetch_hf_models()  # Do expensive operations once
model_dropdown = gr.Dropdown(models)

# ❌ Don't reload in every function
def update_models():
    return gr.Dropdown(choices=fetch_hf_models())  # Expensive!
```

Remember: **Gradio UIs should feel intentional and guide users naturally through complex workflows.**
