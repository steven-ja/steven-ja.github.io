---
title: Electromagnetism
weight: 10
menu:
  notes:
    name: Electromagnetism
    identifier: notes-physics-em
    parent: notes-physics
    weight: 10
---
<!-- A Sample Program -->
{{< note title="Maxwell Equations (Integral)">}}
1. **Gauss' Law**:

$$ \iint_{\partial \Omega} \mathbf{E} \cdot d\mathbf{S} = 4 \pi \iiint_{\Omega} \rho dV $$ 

2. **Gauss' Law for Magnetism**:
$$ \iint_{\partial \Omega} \mathbf{B} \cdot d\mathbf{S} = 0 $$

3. **Maxwell-Faraday Equation**:

$$ \oint_{\partial \Omega} \mathbf{E} \cdot d\mathbf{l} = -\frac{d}{dt} \\int_{\Sigma} \mathbf{B} \cdot d\mathbf{S} $$

4. **Ampère's circuital law**:

$$ \oint_{\partial \Omega} \mathbf{B} \cdot d\mathbf{l} = \mu_0 \left(\iint_{\Sigma} \mathbf{J} \cdot d\mathbf{S} + \epsilon_0 \frac{d}{dt} \iint_{\Sigma} \mathbf{E} \cdot d\mathbf{S}\right) $$
{{< /note >}}

{{< note title="Maxwell Equations (Differential)">}}
1. **Gauss' Law**:

$$ \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0} $$ 

2. **Gauss' Law for Magnetism**:

$$ \nabla \cdot \mathbf{B} = 0 $$

3. **Maxwell-Faraday Equation**:

$$ \nabla \times \mathbf{E} = \frac{\partial \mathbf{B}}{\partial t} $$

4. **Ampère's circuital law**:

$$ \nabla \times \mathbf{B} = \mu_0 \left( \mathbf{J} + \epsilon_0 \frac{\partial \mathbf{E}}{\partial t} \right) $$
{{< /note >}}

