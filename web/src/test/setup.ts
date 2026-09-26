import "@testing-library/jest-dom/vitest"
import { cleanup } from "@testing-library/react"
import { afterEach } from "vitest"

// Без globals: true vitest не чистит DOM сам — рендеры копились бы между тестами.
afterEach(cleanup)
