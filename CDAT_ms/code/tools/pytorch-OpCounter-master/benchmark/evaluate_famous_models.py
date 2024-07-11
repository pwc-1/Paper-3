from thop.profile import profile
import mindspore
import x2ms_adapter

model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__") # and "inception" in name
                     and callable(models.__dict__[name]))

print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
print("---|---|---")

device = "cpu"
if x2ms_adapter.is_cuda_available():
    device = "cuda"

for name in model_names:
    model = x2ms_adapter.to(models.__dict__[name](), device)
    dsize = (1, 3, 224, 224)
    if "inception" in name:
        dsize = (1, 3, 299, 299)
    inputs = x2ms_adapter.to(x2ms_adapter.randn(dsize), device)
    total_ops, total_params = profile(model, (inputs,), verbose=False)
    print("%s | %.2f | %.2f" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))
