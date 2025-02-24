using enso_project
using Documenter

DocMeta.setdocmeta!(enso_project, :DocTestSetup, :(using enso_project); recursive=true)

makedocs(;
    modules=[enso_project],
    authors="Lisa, Beer <lisa.beer@tum.de>, Andrea Pinke <andrea.pinke@tum.de>",
    sitename="enso_project.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
