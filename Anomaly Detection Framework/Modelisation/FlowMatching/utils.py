import torch    


# def ode_solve_euler(x1, model, num_steps=100):

#     x = x1.clone()
#     B = x.shape[0]  
#     dt = 1.0 / num_steps
#     total_norm = torch.zeros(B, device=x.device)

#     for i in range(num_steps):
#         t_scalar = i * dt
#         t = torch.full((B, 1), t_scalar, device=x.device, dtype=x.dtype)
#         u = model(x, t)
        
#         total_norm += u.norm(dim=1) * dt
#         x = x + dt * u        

#     return x, total_norm                  

# def backward_flow(x1, flow_model, n_steps=50):

#     dt = 1.0 / n_steps
#     x = x1.clone()

#     for i in range(n_steps):
#         t = 1 - i * dt
#         t_tensor = torch.ones((x.shape[0], 1), device=x.device) * t
#         v = flow_model(x, t_tensor)
#         x = x - v * dt  

#     return x  


# def forward_flow(z, flow_model, n_steps=50):

#     dt = 1.0 / n_steps
#     x = z.clone()

#     for i in range(n_steps):
#         t = i * dt
#         t_tensor = torch.ones((x.shape[0], 1), device=x.device) * t
#         v = flow_model(x, t_tensor)
#         x = x + v * dt 
#     return x 


# def anomaly_score(x, flow_model, n_steps=50, alpha=1.0, beta=1.0):


#     # backward: data -> latent
#     z = backward_flow(x, flow_model, n_steps=n_steps)

#     # forward: latent -> reconstruction
#     x_hat = forward_flow(z, flow_model, n_steps=n_steps)

#     # reconstruction error
#     rec_err = torch.norm(x - x_hat, dim=1)*100  # (B,)

#     # latent gaussian energy
#     latent_norm = (z ** 2).sum(dim=1)
#     # latent_norm = (z).norm(dim=1)

#     score = alpha * rec_err + beta * latent_norm
#     return score, rec_err, latent_norm, z, x_hat

