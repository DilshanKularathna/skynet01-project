from jaclang.runtimelib.machine import plugin_manager
for plugin in plugin_manager.get_plugins():
    print(f"Loaded plugin: {plugin}")
