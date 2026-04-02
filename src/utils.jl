# Tools for the package

"""
    interpolation(xl, ul, xr, ur, quadrature_point, problem)

Interpolate u and ``\\frac{du}{dx}`` between two discretization points at some specific quadrature point.
Method use for scalar PDE.

Input arguments:

  - `xl`: left boundary of the current interval.
  - `ul`: solution evaluated at the left boundary of the current interval.
  - `xr`: right boundary of the current interval.
  - `ur`: solution evaluated at the right boundary of the current interval.
  - `quadrature_point`: quadrature point chosen according to the method described in [1].
  - `problem`: Structure of type [`SkeelBerzins.ProblemDefinition`](@ref).
"""
@inline function interpolation(xl, ul, xr, ur, qd_point, pb)
    if pb.singular
        h = xr^2 - xl^2
        tmp = qd_point^2 - xl^2

        u = ul * (1 - (tmp / h)) + ur * (tmp / h)
        dudx = (2 * qd_point * (ur - ul)) / h

    elseif pb.m == 0
        h = xr - xl

        u = (ul * (xr - qd_point) + ur * (qd_point - xl)) / h
        dudx = (ur - ul) / h

    elseif pb.m == 1
        tmp = log(xr / xl)
        test_fct = log(qd_point / xl) / tmp

        u = ul * (1 - test_fct) + ur * test_fct
        dudx = (ur - ul) / (qd_point * tmp)

    elseif pb.m == 2
        h = xr - xl
        test_fct = (xr * (qd_point - xl)) / (qd_point * h)

        u = ul * (1 - test_fct) + ur * test_fct
        dudx = (ur - ul) * ((xr * xl) / (qd_point * qd_point * h))
    end

    return u, dudx
end

"""
    interpolation(xl, ul, xr, ur, quadrature_point, m, singular, npde)

Interpolate u and ``\\frac{du}{dx}`` between two discretization points at some specific quadrature point
and return them as static vectors. Method use for system of PDEs.

Input arguments:

  - `xl`: left boundary of the current interval.
  - `ul`: solution evaluated at the left boundary of the current interval.
  - `xr`: right boundary of the current interval.
  - `ur`: solution evaluated at the right boundary of the current interval.
  - `quadrature_point`: quadrature point chosen according to the method described in [1].
  - `m`: symmetry of the problem (given as a type).
  - `singular`: indicates whether the problem is regular or singular (given as a type).
  - `npde`: number of PDEs (given as a type).
"""
@inline function interpolation(xl, ul, xr, ur, qd_point, m, ::Val{true}, ::Val{npde}) where {npde}
    h = xr^2 - xl^2
    tmp = qd_point^2 - xl^2

    interp = SVector{npde}(ul[i] * (1 - (tmp / h)) + ur[i] * (tmp / h) for i ∈ 1:npde)
    dinterp = SVector{npde}((2 * qd_point * (ur[i] - ul[i])) / h for i ∈ 1:npde)

    interp, dinterp
end

@inline function interpolation(xl, ul, xr, ur, qd_point, ::Val{0}, ::Val{false}, ::Val{npde}) where {npde}
    h = xr - xl

    interp = SVector{npde}((ul[i] * (xr - qd_point) + ur[i] * (qd_point - xl)) / h for i ∈ 1:npde)
    dinterp = SVector{npde}((ur[i] - ul[i]) / h for i ∈ 1:npde)

    interp, dinterp
end

@inline function interpolation(xl, ul, xr, ur, qd_point, ::Val{1}, ::Val{false}, ::Val{npde}) where {npde}
    tmp = log(xr / xl)
    test_fct = log(qd_point / xl) / tmp

    interp = SVector{npde}(ul[i] * (1 - test_fct) + ur[i] * test_fct for i ∈ 1:npde)
    dinterp = SVector{npde}((ur[i] - ul[i]) / (qd_point * tmp) for i ∈ 1:npde)

    interp, dinterp
end

@inline function interpolation(xl, ul, xr, ur, qd_point, ::Val{2}, ::Val{false}, ::Val{npde}) where {npde}
    h = xr - xl
    test_fct = (xr * (qd_point - xl)) / (qd_point * h)

    interp = SVector{npde}(ul[i] * (1 - test_fct) + ur[i] * test_fct for i ∈ 1:npde)
    dinterp = SVector{npde}((ur[i] - ul[i]) * ((xr * xl) / (qd_point * qd_point * h)) for i ∈ 1:npde)

    interp, dinterp
end

"""
$(TYPEDEF)

Structure storing the problem definition.

$(TYPEDFIELDS)
"""
struct ProblemDefinition{T1, T2, T3, Tv <: AbstractVector, Ti <: Integer, Tm <: Number, elTv <: Number,
                         pdeFunction <: Function,
                         icFunction <: Function,
                         bdFunction <: Function}
    """
    Number of unknowns
    """
    npde::Ti

    """
    Number of discretization points
    """
    Nx::Ti

    """
    Grid of the problem
    """
    xmesh::Tv

    """
    Time interval
    """
    tspan::Tuple{Tm, Tm}

    """
    Flag to know if the problem is singular or not
    """
    singular::Bool

    """
    Symmetry of the problem
    """
    m::Ti

    """
    Jacobi matrix
    """
    # jac::Union{SparseMatrixCSC{elTv, Ti}, BandedMatrix{elTv, Matrix{elTv}, Base.OneTo{Ti}}}
    jac::SparseMatrixCSC{elTv, Ti}

    """
    Number of design variables for PDE Constrained Optimization
    """
    nb_design_var::Ti

    """
    Evaluation of the initial condition
    """
    inival::Vector{elTv}

    """
    Interpolation points from the paper
    """
    ξ::Vector{elTv}
    ζ::Vector{elTv}

    """
    Function defining the coefficients of the PDE
    """
    pdefunction::pdeFunction

    """
    Function defining the initial condition
    """
    icfunction::icFunction

    """
    Function defining the boundary conditions
    """
    bdfunction::bdFunction
end

"""
$(TYPEDEF)

Structure containing all the keyword arguments for the solver [`pdepe`](@ref).

$(TYPEDFIELDS)
"""
Base.@kwdef struct Params
    """
    Choice of the time discretization either use `:euler` for internal implicit Euler method or `:DiffEq` for the [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) package.
    """
    solver::Symbol = :euler

    """
    Defines a time step (either pass a `Float64` or a `Vector`) when using the implicit Euler method.
    When set to `tstep=Inf`, it solves the stationary version of the problem.
    """
    tstep::Union{Float64, Vector{Float64}} = 1e-2

    """
    Flag, returns with the solution, a list of 1d-array with the history from the newton solver.
    """
    hist::Bool = false

    """
    Choice of the type of matrix (`:sparseArrays`, `:banded`) use to store the jacobian.
    """
    sparsity::Symbol = :sparseArrays

    """
    Choice of the solver for the LSE in the newton method, see [`LinearSolve.jl`](https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/).
    """
    linsolve::Union{LinearSolve.SciMLLinearSolveAlgorithm, Nothing} = KLUFactorization()

    """
    Maximum number of iterations for the Newton solver.
    """
    maxit::Int = 100

    """
    Tolerance used for Newton method.
    Returns solution if ``||\\; u_{i+1} - u_{i} \\;||_2 <`` `tol`.
    """
    tol::Float64 = 1e-10

    """
    Returns the data of the PDE problem
    """
    data::Bool = false

    """
    Number of design variables for PDE Constrained Optimization
    """
    nb_design_var::Int = 0
end

"""
    problem_init(m, xmesh, tspan, pdefun, icfun, bdfun, params)

Function initializing the problem.

Input arguments: similar as [`pdepe`](@ref).

Returns the size of the space discretization Nx, the number of PDEs npde, the initial value inival,
some data types elTv and Ti, and the struct containing the problem definition pb.
"""
function problem_init(m, xmesh, tspan, pdefun::T1, icfun::T2, bdfun::T3, params) where {T1, T2, T3}

    # Size of the space discretization
    Nx = length(xmesh)

    # Regular case: m=0 or a>0 (Galerkin method)
    # Singular case: m≥1 and a=0 (Petrov-Galerkin method)
    singular = m ≥ 1 && xmesh[1] == 0

    α = @view xmesh[1:(end - 1)]
    β = @view xmesh[2:end]
    γ = (α .+ β) ./ 2

    ξ, ζ = get_quad_points_weights(m, α, β, γ, singular)

    Tv = typeof(xmesh)
    elTv = eltype(xmesh)
    Tm = eltype(tspan)

    # Number of unknows in the PDE problem
    npde = length(icfun(xmesh[1]))

    # Reshape inival as a 1D array for compatibility with the solvers from DifferentialEquations.jl
    inival = npde == 1 ? icfun.(xmesh) : vec(reduce(hcat, icfun.(xmesh)))

    Ti = eltype(npde)

    # jac = get_sparsity_pattern(SparseMatrixCSC{elTv, Ti}, Nx, npde, elTv)

    pb = ProblemDefinition{m, npde, singular, Tv, Ti, Tm, elTv, T1, T2, T3}(npde,
                                                                            Nx,
                                                                            xmesh,
                                                                            tspan,
                                                                            singular,
                                                                            m,
                                                                            # jac,
                                                                            params.nb_design_var,
                                                                            inival,
                                                                            ξ,
                                                                            ζ,
                                                                            pdefun,
                                                                            icfun,
                                                                            bdfun)

    Nx, npde, inival, elTv, Ti, pb
end

Base.length(pb::SkeelBerzins.ProblemDefinition) = pb.nb_design_var

"""
    get_quad_points_weights(m, alpha, beta, gamma, singular)

Calculate the quadrature points and weights for the one-point Gauss quadrature based on the
problem's specific symmetry, as described in the paper [1].

Input arguments:

  - `m`: scalar representing the symmetry of the problem.
  - `alpha`: 1D array containing the left boundaries of the subintervals.
  - `beta`: 1D array containing the right boundaries of the subintervals.
  - `gamma`: 1D array containing the middle points of the subintervals.
  - `singular`: boolean indicating whether the problem is singular or not.

Returns the quadrature points `xi` and the weights `zeta`.
"""
function get_quad_points_weights(m, α, β, γ, singular)

    # Cartesian Coordinates
    if m == 0 # (Always Regular case)

        # Quadrature point ξ and weight ζ for m=0
        ξ = γ
        ζ = γ

        # Cylindrical Polar Coordinates
    elseif m == 1

        # Quadrature point ξ and weight ζ for m=1
        if singular
            ξ = (2 / 3) .* (α .+ β .- ((α .* β) ./ (α .+ β)))
            ζ = ((β .^ 2 .- α .^ 2) ./ (2 .* log.(β ./ α))) .^ (0.5)
        else # Regular case
            ξ = (β .- α) ./ log.(β ./ α)
            ζ = (ξ .* γ) .^ (0.5)
        end

        # Spherical Polar Coordinates
    else
        m == 2

        # Quadrature point ξ and weight ζ for m=2
        if singular
            ξ = (2 / 3) .* (α .+ β .- ((α .* β) ./ (α .+ β)))
        else # Regular case
            ξ = (α .* β .* log.(β ./ α)) ./ (β .- α)
        end
        ζ = (α .* β .* γ) .^ (1 / 3)
    end

    ξ, ζ
end

"""
    get_sparsity_pattern(sparsity, Nx, npde, elTv)

Function that provides the sparsity pattern in a SparseMatrixCSC.
"""
function get_sparsity_pattern(sparsity::Type{TMat},
                              Nx,
                              npde,
                              elTv) where {TMat <: SparseArrays.AbstractSparseMatrixCSC}
    row = Int64[]
    column = Int64[]
    vals = elTv[]

    for i ∈ 1:npde
        for j ∈ 1:(2 * npde)
            push!(row, i)
            push!(column, j)
            push!(vals, one(elTv))
        end
    end
    for i ∈ ((Nx - 1) * npde + 1):(Nx * npde)
        for j ∈ ((Nx - 2) * npde + 1):(Nx * npde)
            push!(row, i)
            push!(column, j)
            push!(vals, one(elTv))
        end
    end
    for i ∈ (npde + 1):npde:((Nx - 1) * npde)
        for k ∈ i:(i + npde - 1)
            for j ∈ (i - npde):(i + 2 * npde - 1)
                push!(row, k)
                push!(column, j)
                push!(vals, one(elTv))
            end
        end
    end
    jac = sparse(row, column, vals)

    jac
end
